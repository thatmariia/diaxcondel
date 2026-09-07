"""
Controls generated from catalog metadata.

Every control in the playground is derived from a
:class:`~diaxcondel.catalog.param.ParamField`: its type picks the widget, its
bounds pick the range, its documented description becomes the help text. A
component added to the catalog therefore appears in the dashboard, fully
labelled, with no changes here.

Only the options of the *selected* component are ever shown, and options
marked advanced stay folded away until asked for, so the page shows what the
current choice actually needs, not the union of everything it could need.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from diaxcondel.catalog import ComponentSpec, ParamField, options
from diaxcondel.catalog.registry import Slot
from diaxcondel.connectome.manual import format_matrix, parse_codes, parse_matrix
from diaxcondel.experiment import ComponentRef

_OTHER = "something else…"


def _resize(matrix: np.ndarray, n: int) -> np.ndarray:
    """Pad with zeros or crop a square matrix to size ``n``."""
    out = np.zeros((n, n))
    k = min(n, matrix.shape[0])
    out[:k, :k] = matrix[:k, :k]
    return out


def _matrix_widget(
    field: ParamField,
    key: str,
    value: Any,
    labels: Sequence[str],
    siblings: Mapping[str, Any],
) -> str:
    """Render an editable table for a matrix-valued text field."""
    try:
        matrix = parse_matrix(str(value or ""), n=len(labels) or None)
    except ValueError:
        matrix = np.zeros((max(2, len(labels)), max(2, len(labels))))

    row_labels = list(labels)
    if not row_labels:
        sibling_codes = str(siblings.get("codes", "") or "")
        size = int(
            st.number_input(
                "number of regions",
                min_value=2,
                max_value=64,
                value=int(matrix.shape[0]),
                key=f"{key}__size",
                help="Rows and columns of the table below. Growing it adds zeros; shrinking it crops.",
            )
        )
        if size != matrix.shape[0]:
            matrix = _resize(matrix, size)
        try:
            row_labels = list(parse_codes(sibling_codes, n=size))
        except ValueError:
            row_labels = [f"R{i + 1}" for i in range(size)]
    elif matrix.shape[0] != len(row_labels):
        matrix = _resize(matrix, len(row_labels))
        st.caption(f"Table resized to the model's {len(row_labels)} regions.")

    if field.description:
        st.caption(f"{field.description} Rows are targets, columns are sources.")
    frame = pd.DataFrame(matrix, index=list(row_labels), columns=list(row_labels))
    edited = st.data_editor(
        frame,
        key=key,
        width="stretch",
        column_config={code: st.column_config.NumberColumn(code, format="%.3f") for code in row_labels},
    )
    return format_matrix(np.asarray(pd.DataFrame(edited).to_numpy(), dtype=float))


def _number_widget(field: ParamField, key: str, value: Any) -> Any:
    """Render an int/float control, as a slider when the range is bounded."""
    is_int = field.kind == "int"
    step = field.step if field.step is not None else (1 if is_int else None)
    if field.minimum is not None and field.maximum is not None:
        cast = int if is_int else float
        clamped = min(max(value, field.minimum), field.maximum)
        return st.slider(
            field.label,
            min_value=cast(field.minimum),
            max_value=cast(field.maximum),
            value=cast(clamped),
            step=cast(step) if step is not None else None,
            key=key,
            help=field.description or None,
        )
    return st.number_input(
        field.label,
        value=int(value) if is_int else float(value),
        step=step,
        min_value=field.minimum,
        max_value=field.maximum,
        key=key,
        help=field.description or None,
    )


def render_field(
    field: ParamField,
    key: str,
    value: Any,
    *,
    region_codes: Sequence[str] = (),
    siblings: Mapping[str, Any] | None = None,
    pool: Sequence[str] | None = None,
) -> Any:
    """
    Render one option and return the chosen value.

    Parameters
    ----------
    field : ParamField
        Field metadata from the catalog.
    key : str
        Unique widget key.
    value : object
        Current value.
    region_codes : sequence of str
        Region codes offered for fields marked with the ``"regions"`` widget,
        and used to label matrix editors of connectivity entries.
    siblings : Mapping[str, Any], optional
        Values of the entry's other options already rendered, so a control
        can adapt to them (a matrix editor uses a sibling list of codes).
    pool : sequence of str, optional
        Options to offer instead of the ones declared in the catalog, for
        choices that only the caller can know, such as the parts of the brain a
        particular atlas actually has, for instance.

    Returns
    -------
    object
        The value selected by the user.
    """
    siblings = siblings or {}

    if field.optional:
        prompt = (
            f"{field.label}: keep every region (no merging)"
            if field.name == "merge_level"
            else f"{field.label}: leave unset"
        )
        unset = st.checkbox(prompt, value=value is None, key=f"{key}__none", help=field.description or None)
        if unset:
            return None
        if value is None:
            value = field.default if field.default is not None else (0 if field.kind == "int" else 0.0)

    if field.widget == "matrix":
        own_regions = field.name == "matrix_cm"  # a distance table defines its own regions
        return _matrix_widget(field, key, value, () if own_regions else region_codes, siblings)

    if field.widget == "vector":
        return st.text_area(field.label, value=str(value), key=key, help=field.description or None, height=80)

    if field.kind == "bool":
        return st.checkbox(field.label, value=bool(value), key=key, help=field.description or None)

    if field.kind == "str_list":
        if pool is not None:
            options_pool = [str(item) for item in pool]
        elif field.widget == "regions":
            options_pool = list(region_codes)
        else:
            options_pool = [str(c) for c in (field.choices or ())]
        if field.open_choices:
            options_pool = list(dict.fromkeys([*options_pool, *(str(item) for item in (value or ()))]))
        selected = [item for item in (value or ()) if item in options_pool]
        return tuple(
            st.multiselect(
                field.label,
                options=options_pool,
                default=selected,
                key=key,
                help=field.description or None,
                placeholder="all regions" if field.widget == "regions" else "everything",
            )
        )

    if field.choices:
        choices = [str(c) for c in field.choices]
        if field.open_choices:
            choices = [*choices, _OTHER]
        current = str(value)
        index = choices.index(current) if current in choices else (len(choices) - 1 if field.open_choices else 0)
        picked = st.selectbox(
            field.label,
            options=choices,
            index=index,
            format_func=lambda option: "(none)" if option == "" else option,
            key=key,
            help=field.description or None,
        )
        if picked == _OTHER:
            picked = st.text_input(
                f"{field.label}: type a value",
                value="" if current in choices else current,
                key=f"{key}__custom",
            ) or str(field.default)
        return field.coerce(picked)

    if field.kind in {"int", "float"}:
        return _number_widget(field, key, value)

    return st.text_input(field.label, value=str(value), key=key, help=field.description or None)


def render_entry_params(
    entry: ComponentSpec,
    params: Mapping[str, Any],
    *,
    key_prefix: str,
    region_codes: Sequence[str] = (),
    columns: int = 1,
    pools: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """
    Render every option of a catalog entry, hiding advanced ones by default.

    Parameters
    ----------
    entry : ComponentSpec
        The entry whose options to render.
    params : Mapping[str, Any]
        Current values.
    key_prefix : str
        Prefix for widget keys, unique per position in the form.
    region_codes : sequence of str
        Region codes for region-valued fields and matrix labels.
    columns : int
        Lay the ordinary options out in this many columns.
    pools : Mapping[str, sequence of str], optional
        Option lists supplied by the caller, keyed by field name.

    Returns
    -------
    dict
        Validated parameter mapping.
    """
    pools = dict(pools or {})
    values: dict[str, Any] = {}
    # Fields marked hidden get a control elsewhere on the page; keep their
    # current value so the round trip through the form does not reset them.
    for field in entry.fields:
        if field.widget == "hidden":
            values[field.name] = params.get(field.name, field.default)
    main = [f for f in entry.fields if not f.advanced and f.widget != "hidden"]
    advanced = [f for f in entry.fields if f.advanced and f.widget != "hidden"]

    wide = {"matrix", "vector", "regions", "multiselect"}
    grid = [f for f in main if f.widget not in wide]
    full_width = [f for f in main if f.widget in wide]

    if columns > 1 and len(grid) > 1:
        slots = st.columns(columns)
        for index, field in enumerate(grid):
            with slots[index % columns]:
                values[field.name] = render_field(
                    field,
                    key=f"{key_prefix}.{field.name}",
                    value=params.get(field.name, field.default),
                    region_codes=region_codes,
                    siblings=values,
                    pool=pools.get(field.name),
                )
    else:
        for field in grid:
            values[field.name] = render_field(
                field,
                key=f"{key_prefix}.{field.name}",
                value=params.get(field.name, field.default),
                region_codes=region_codes,
                siblings=values,
                pool=pools.get(field.name),
            )

    for field in full_width:
        values[field.name] = render_field(
            field,
            key=f"{key_prefix}.{field.name}",
            value=params.get(field.name, field.default),
            region_codes=region_codes,
            siblings=values,
            pool=pools.get(field.name),
        )

    if advanced:
        with st.expander(f"Advanced — {len(advanced)} more setting{'s' if len(advanced) > 1 else ''}"):
            for field in advanced:
                values[field.name] = render_field(
                    field,
                    key=f"{key_prefix}.{field.name}",
                    value=params.get(field.name, field.default),
                    region_codes=region_codes,
                    siblings=values,
                    pool=pools.get(field.name),
                )
    return entry.coerce(values)


def render_slot(
    slot: Slot,
    current: ComponentRef | None,
    *,
    key_prefix: str,
    label: str,
    region_codes: Sequence[str] = (),
    columns: int = 1,
    show_reference: bool = True,
    unsuitable: Mapping[str, str] | None = None,
    pools: Mapping[str, Sequence[str]] | None = None,
) -> ComponentRef:
    """
    Render an entry chooser plus the options of the chosen entry.

    Parameters
    ----------
    slot : str
        Catalog slot.
    current : ComponentRef or None
        Currently selected reference, used for defaults.
    key_prefix : str
        Prefix for widget keys.
    label : str
        Heading shown above the chooser.
    region_codes : sequence of str
        Region codes for region-valued fields.
    columns : int
        Column count for the option grid.
    show_reference : bool
        Show the entry's literature citation, when it has one.
    unsuitable : Mapping[str, str], optional
        Entry names that do not fit the rest of the current model, mapped to
        the reason. They stay listed, since knowing an option exists and why
        it does not apply is more useful than hiding it, but say so, and are
        moved to the end of the list.
    pools : Mapping[str, sequence of str], optional
        Option lists supplied by the caller, keyed by field name.

    Returns
    -------
    ComponentRef
        The selected entry and its parameters.

    Raises
    ------
    RuntimeError
        If the slot has no registered entries.
    """
    entries = options(slot)
    if not entries:  # pragma: no cover - only if a slot is emptied
        raise RuntimeError(f"no components registered for slot {slot!r}")

    unsuitable = dict(unsuitable or {})
    ordered = sorted(entries, key=lambda entry: (entry.name in unsuitable, not entry.available))
    names = [entry.name for entry in ordered]
    labels = {}
    for entry in ordered:
        if not entry.available:
            labels[entry.name] = f"⛔ {entry.label} — needs {', '.join(entry.requires)}"
        elif entry.name in unsuitable:
            labels[entry.name] = f"⚠ {entry.label} — does not fit this model"
        else:
            labels[entry.name] = entry.label
    default_index = names.index(current.name) if current and current.name in names else 0
    chosen = st.selectbox(
        label,
        options=names,
        index=default_index,
        format_func=lambda name: labels[name],
        key=f"{key_prefix}.name",
    )
    entry = next(e for e in ordered if e.name == chosen)
    if entry.summary:
        st.caption(entry.summary)
    if not entry.available:
        st.warning(f"Not installed. Run `{entry.install_hint}` to enable it.")
    elif chosen in unsuitable:
        st.warning(unsuitable[chosen])
    if show_reference and entry.reference:
        st.caption(f"Source: {entry.reference}")

    params = current.params if current and current.name == chosen else {}
    values = render_entry_params(
        entry,
        params,
        key_prefix=f"{key_prefix}.{chosen}",
        region_codes=region_codes,
        columns=columns,
        pools=pools,
    )
    return ComponentRef(name=chosen, params=values)
