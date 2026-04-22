from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.types import Schema


def build_default_constraints(
    df: pd.DataFrame,
    schema: Schema,
    include_conditional_blanks: bool = False,
    derived_repair_mode: str = "overwrite",
) -> dict:
    constraints = {
        "ranges": {},
        "allowed_values": {},
        "derived": [],
        "conditional_blanks": [],
        "impossible_combinations": [],
        "repair_options": {
            "derived_repair_mode": str(derived_repair_mode),
        },
    }

    for column in schema.columns:
        series = df[column.name]
        if column.kind == "numeric":
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if not numeric.empty:
                constraints["ranges"][column.name] = [float(numeric.min()), float(numeric.max())]
        elif column.kind in {"categorical", "binary"} and column.unique_count <= 100:
            constraints["allowed_values"][column.name] = list(
                series.dropna().astype("object").drop_duplicates().tolist()
            )

    _add_bmi_rule(constraints, df.columns)
    _add_age_rule(constraints, df.columns)
    _add_postoperative_date_rules(constraints, df)
    if include_conditional_blanks:
        _add_conditional_blank_rules(constraints, df.columns)
    return constraints


def _add_bmi_rule(constraints: dict, columns: pd.Index) -> None:
    weight_col = "Preoperative body weight (kg)::20"
    height_col = "Height (cm)::23"
    bmi_col = "BMI::24"
    if all(column in columns for column in [weight_col, height_col, bmi_col]):
        constraints["derived"].append(
            {
                "target": bmi_col,
                "inputs": [weight_col, height_col],
                "tolerance": 2.0,
                "fn": lambda df: pd.to_numeric(df[weight_col], errors="coerce")
                / ((pd.to_numeric(df[height_col], errors="coerce") / 100.0) ** 2),
            }
        )


def _add_age_rule(constraints: dict, columns: pd.Index) -> None:
    age_col = "Age::40"
    if age_col in columns:
        constraints["ranges"][age_col] = [0.0, 120.0]


def _add_postoperative_date_rules(constraints: dict, df: pd.DataFrame) -> None:
    columns = df.columns
    operation_date_col = "Date of primary operation (YYYY-MM-DD)::53"
    if operation_date_col not in columns:
        return

    rule_specs = [
        ("Termination of intravenous fluid infusion (YYYY-MM-DD)::108", "Duration of IV fluid infusion (nights)::109"),
        ("First passage of flatus (YYYY-MM-DD)::127", "Time to passage of flatus (nights)::129"),
        ("First passage of stool (YYYY-MM-DD)::130", "Time to passage of stool (nights)::131"),
        ("Tolerating solid food (YYYY-MM-DD)::132", "Time to tolerating solid food (nights)::133"),
        ("Termination of urinary drainage (YYYY-MM-DD)::140", "Time to termination of urinary drainage (nights)::141"),
        ("Nursed back to preoperative ADL ability (YYYY-MM-DD)::142", "Time to recovery of ADL ability (nights)::143"),
        ("Termination of epidural analgesia (YYYY-MM-DD)::147", "Time to termination of epidural analgesia (nights)::149"),
        ("Pain control adequate on oral analgesics (YYYY-MM-DD)::155", "Time to pain control with oral analgesics (nights)::156"),
        ("Date of discharge (YYYY-MM-DD)::178", "Length of stay (nights in hospital after primary operation)::179"),
        ("Date of follow-up (YYYY-MM-DD)::232", "Time between operation and follow-up (nights)::235"),
    ]

    for target_col, duration_col in rule_specs:
        if target_col not in columns or duration_col not in columns:
            continue
        constraints["derived"].append(
            {
                "target": target_col,
                "inputs": [operation_date_col, duration_col],
                "kind": "datetime",
                "tolerance": 1.0,
                "target_missing_rate": float(df[target_col].isna().mean()),
                "fn": lambda df, base=operation_date_col, duration=duration_col: _offset_date_strings(
                    df[base],
                    df[duration],
                ),
            }
        )


def _offset_date_strings(
    base_dates: pd.Series,
    durations: pd.Series,
) -> pd.Series:
    base = pd.to_datetime(base_dates, errors="coerce")
    numeric_durations = pd.to_numeric(durations, errors="coerce").round()
    offset = pd.to_timedelta(numeric_durations, unit="D")
    shifted = base + offset
    formatted = shifted.dt.strftime("%Y-%m-%d")
    valid = base.notna() & numeric_durations.notna()
    return formatted.where(valid, np.nan)


def _add_conditional_blank_rules(constraints: dict, columns: pd.Index) -> None:
    primary_subgroups = [
        "Respiratory complication(s)::189",
        "Infectious complication(s)::197",
        "Cardiovascular complication(s)::205",
        "Renal, hepatic, pancreatic and gastrointestinal complication(s)::215",
        "Surgical complication(s)::230",
        "Complication(s) related to epidural or spinal anaesthesia::246",
        "Anaesthetic complication(s)::250",
    ]
    _append_conditional_blank_rule(
        constraints,
        columns,
        parent="Complications at all during primary stay::183",
        children=primary_subgroups,
        inactive_prefixes=["No"],
    )
    _append_conditional_blank_rule(
        constraints,
        columns,
        parent="Respiratory complication(s)::189",
        children=[
            "Lobar atelectasis::190",
            "Pneumonia::191",
            "Pleural Fluid::192",
            "Respiratory failure::193",
            "Pneumothorax::194",
            "Other respiratory complication::195",
        ],
        inactive_prefixes=["No"],
    )
    _append_conditional_blank_rule(
        constraints,
        columns,
        parent="Infectious complication(s)::197",
        children=[
            "Wound Infection::204",
            "Urinary tract infection::203",
            "Intraperitoneal or retroperitoneal abscess::202",
            "Sepsis::201",
            "Septic Shock::200",
            "Infected graft or prosthesis::199",
            "Other infectious complication::198",
        ],
        inactive_prefixes=["No"],
    )
    _append_conditional_blank_rule(
        constraints,
        columns,
        parent="Cardiovascular complication(s)::205",
        children=[
            "Heart Failure::214",
            "Acute Myocardial Infarction::213",
            "Deep Venous Thrombosis::212",
            "Portal Vein Thrombosis::211",
            "Pulmonary Embolus::210",
            "Cerebrovascular lesion::209",
            "Cardiac arrhythmia::208",
            "Cardiac arrest::207",
            "Other cardiovascular complication::206",
        ],
        inactive_prefixes=["No"],
    )
    _append_conditional_blank_rule(
        constraints,
        columns,
        parent="Renal, hepatic, pancreatic and gastrointestinal complication(s)::215",
        children=[
            "Renal dysfunction::228",
            "Urinary retention::226",
            "Hepatic dysfunction::225",
            "Pancreatitis::220",
            "Gastrointestinal haemorrhage::219",
            "Nausea or vomiting::218",
            "Obstipation or diarrhoea::217",
            "Other organ dysfunction::216",
        ],
        inactive_prefixes=["No"],
    )
    _append_conditional_blank_rule(
        constraints,
        columns,
        parent="Surgical complication(s)::230",
        children=[
            "Anastomotic leak::244",
            "Urinary tract injury::243",
            "Mechanical bowel obstruction::241",
            "Postoperative paralytic ileus::240",
            "Deep wound dehiscence::239",
            "Intraoperative excessive haemorrhage::237",
            "Postoperative excessive haemorrhage::236",
            "Other surgical technical complication or injury::234",
            "Hematoma::233",
        ],
        inactive_prefixes=["No"],
    )
    _append_conditional_blank_rule(
        constraints,
        columns,
        parent="Complication(s) related to epidural or spinal anaesthesia::246",
        children=[
            "Post dural-puncture headache::249",
            "Epidural hematoma or abscess::248",
            "Other EDA or spinal related complication::247",
        ],
        inactive_prefixes=["No"],
    )
    _append_conditional_blank_rule(
        constraints,
        columns,
        parent="Anaesthetic complication(s)::250",
        children=[
            "Pulmonary aspiration of gastric contents::257",
            "Hypotension::256",
            "Hypoxia::255",
            "Prolonged postoperative sedation::251",
            "Other anaesthetic complication(s)::253",
        ],
        inactive_prefixes=["No"],
    )

    followup_subgroups = [
        "Respiratory complication(s)::297",
        "Infectious complication(s)::312",
        "Cardiovascular complication(s)::282",
        "Renal, hepatic, pancreatic and gastrointestinal complication(s)::298",
        "Surgical complication(s)::325",
        "Complication(s) related to epidural or spinal anaesthesia::326",
        "Anaesthetic complication(s)::331",
    ]
    _append_conditional_blank_rule(
        constraints,
        columns,
        parent="Complications at all after primary stay::283",
        children=followup_subgroups,
        inactive_prefixes=["No"],
    )
    _append_conditional_blank_rule(
        constraints,
        columns,
        parent="Respiratory complication(s)::297",
        children=[
            "Lobar atelectasis::300",
            "Pneumonia::301",
            "Pleural Fluid::305",
            "Respiratory failure::308",
            "Pneumothorax::307",
            "Other respiratory complication::303",
        ],
        inactive_prefixes=["No"],
    )
    _append_conditional_blank_rule(
        constraints,
        columns,
        parent="Infectious complication(s)::312",
        children=[
            "Wound Infection::323",
            "Urinary tract infection::320",
            "Intraperitoneal or retroperitoneal abscess::317",
            "Sepsis::319",
            "Septic Shock::318",
            "Infected graft or prosthesis::314",
            "Other infectious complication::315",
        ],
        inactive_prefixes=["No"],
    )
    _append_conditional_blank_rule(
        constraints,
        columns,
        parent="Cardiovascular complication(s)::282",
        children=[
            "Heart failure::287",
            "Acute myocardial infarction::288",
            "Deep venous thrombosis::285",
            "Portal Vein Thrombosis::289",
            "Pulmonary embolus::291",
            "Cerebrovascular lesion::294",
            "Cardiac arrhythmia::295",
            "Cardiac arrest::296",
            "Hypertension::316",
            "Other cardiovascular complication::292",
        ],
        inactive_prefixes=["No"],
    )
    _append_conditional_blank_rule(
        constraints,
        columns,
        parent="Renal, hepatic, pancreatic and gastrointestinal complication(s)::298",
        children=[
            "Renal dysfunction::299",
            "Urinary retention::352",
            "Hepatic dysfunction::302",
            "Pancreatitis::304",
            "Gastrointestinal haemorrhage::306",
            "Nausea or vomiting::310",
            "Obstipation or diarrhoea::311",
            "Incontinence::313",
            "Other organ dysfunction::309",
        ],
        inactive_prefixes=["No"],
    )
    _append_conditional_blank_rule(
        constraints,
        columns,
        parent="Surgical complication(s)::325",
        children=[
            "Anastomotic leak::324",
            "Urinary tract injury::328",
            "Mechanical bowel obstruction::322",
            "Postoperative paralytic ileus::321",
            "Deep wound dehiscence::340",
            "Intraoperative excessive haemorrhage::339",
            "Postoperative excessive haemorrhage::338",
            "Other surgical technical complication or injury::337",
            "Hematoma::336",
        ],
        inactive_prefixes=["No"],
    )
    _append_conditional_blank_rule(
        constraints,
        columns,
        parent="Complication(s) related to epidural or spinal anaesthesia::326",
        children=[
            "Post dural-puncture headache::327",
            "Epidural hematoma or abscess::329",
            "Other EDA or spinal related complication::330",
        ],
        inactive_prefixes=["No"],
    )
    _append_conditional_blank_rule(
        constraints,
        columns,
        parent="Anaesthetic complication(s)::331",
        children=[
            "Pulmonary aspiration of gastric contents::257",
            "Hypotension::256",
            "Hypoxia::255",
            "Prolonged postoperative sedation::251",
        ],
        inactive_prefixes=["No"],
    )

    _append_conditional_blank_rule(
        constraints,
        columns,
        parent="Anastomosis::66",
        children=["Type of anastomosis::67", "Anastomotic technique::68"],
        inactive_prefixes=["No"],
    )
    _append_conditional_blank_rule(
        constraints,
        columns,
        parent="Follow-up performed::231",
        children=[
            "Date of follow-up (YYYY-MM-DD)::232",
            "Time between operation and follow-up (nights)::235",
            "WHO Performance Score at follow-up::238",
        ],
        inactive_prefixes=["No"],
    )
    _append_conditional_blank_rule(
        constraints,
        columns,
        parent="Readmission(s)::280",
        children=["Length of stay for readmissions::354"],
        inactive_prefixes=["No"],
    )
    _append_conditional_blank_rule(
        constraints,
        columns,
        parent="Diabetes Mellitus::11",
        children=[
            "Last HbA1c value ((mmol/mol))::28",
            "Last HbA1c value (Unknown)::28",
        ],
        inactive_prefixes=["No"],
    )
    _append_conditional_blank_rule(
        constraints,
        columns,
        parent="Postoperative epidural analgesia::145",
        children=[
            "Time to termination of epidural analgesia (nights)::149",
            "Successful block?::150",
        ],
        inactive_prefixes=["No"],
    )
    _append_conditional_blank_rule(
        constraints,
        columns,
        parent="Epidural or spinal anaesthesia::88",
        children=["Level of insertion::89"],
        inactive_prefixes=["No"],
    )


def _append_conditional_blank_rule(
    constraints: dict,
    columns: pd.Index,
    parent: str,
    children: list[str],
    inactive_prefixes: list[str] | None = None,
    inactive_values: list[str] | None = None,
) -> None:
    if parent not in columns:
        return

    available_children = [column for column in children if column in columns]
    if not available_children:
        return

    constraints["conditional_blanks"].append(
        {
            "parent": parent,
            "children": available_children,
            "inactive_prefixes": inactive_prefixes or [],
            "inactive_values": inactive_values or [],
        }
    )
