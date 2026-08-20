"""Create clean CSV and multi-sheet Excel exports for completed ablations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from string import ascii_uppercase
from typing import Any, Sequence
from zipfile import ZIP_DEFLATED, ZipFile
from xml.sax.saxutils import escape

from analyze_results import (
    ALL_METRICS,
    EXTRA_COLUMNS,
    check_complete,
    check_results,
    load_all,
)


TECHNICAL_COLUMNS = (
    "protocol_version", "mi_quantization", "warp_backend",
)
SUMMARY_COLUMNS = tuple(
    column for column in (
        "case", "crop_id", "seed", "k1_true", "k2_true", "noise", "method",
        "status",
    ) + ALL_METRICS + EXTRA_COLUMNS
    if column not in TECHNICAL_COLUMNS
)
CSV_COLUMNS = ("ablation",) + SUMMARY_COLUMNS

# The compact exports keep the measurements needed for comparing accuracy,
# image quality, and runtime while omitting provenance and secondary setup
# fields.  The detailed combined CSV above retains the other result metrics.
Record = dict[str, Any]

READABLE_COLUMNS = (
    "case", "crop_id", "seed", "k1_true", "noise", "method", "status",
    "k1_err", "k1_signed", "pos_rmse", "pos_mae", "pos_max",
    "seam_true", "seam_est", "psnr", "ssim", "ncc", "mse",
    "coverage_fraction", "distortion_psnr", "distortion_ssim",
    "distortion_ncc", "distortion_mse", "distortion_coverage_fraction",
    "time_s", "iterations",
)

# Use plain-language labels in the Excel workbook while retaining the
# canonical method names in the raw JSON and CSV exports.
EXCEL_SHEET_NAMES = {
    "oracle_k1": "Known k1",
    "joint_oracle": "Joint known k1",
    "k1_magnitude": "K1 magnitude",
}
EXCEL_METHOD_NAMES = {
    "oracle_k1_paper": "Known k1",
    "dcs_paper_joint_oracle": "Joint, known k1",
}

# Each group contains the ablation arms plus the reference arm used by the
# existing per-directory analyses.  Keeping the references in the combined
# file makes each group directly comparable without duplicating primary data
# in the ablation result directories themselves.
ABLATION_SPECS = (
    {
        "name": "feedback_iterations",
        "results": Path("results/ablation_feedback"),
        "reference": Path("results/distortcorrect"),
        "methods": (
            "dcs_paper_iter_1", "dcs_paper_iter_2",
            "dcs_paper_iter_5", "dcs_paper_iter_10",
            "dcs_paper_style",
        ),
        "ablation_methods": (
            "dcs_paper_iter_1", "dcs_paper_iter_2",
            "dcs_paper_iter_5", "dcs_paper_iter_10",
        ),
        "reference_methods": ("dcs_paper_style",),
        "cases": "all",
    },
    {
        "name": "prestitch",
        "results": Path("results/ablation_prestitch"),
        "reference": (
            Path("results/distortcorrect"),
            Path("results/sequential"),
        ),
        "methods": (
            "dcs_paper_no_prestitch", "sequential_paper_no_prestitch",
            "dcs_paper_style", "sequential_paper_matched",
        ),
        "ablation_methods": (
            "dcs_paper_no_prestitch", "sequential_paper_no_prestitch",
        ),
        "reference_methods": ("dcs_paper_style", "sequential_paper_matched"),
        "cases": "all",
    },
    {
        "name": "search_range",
        "results": Path("results/ablation_search"),
        "reference": Path("results/sequential"),
        "methods": (
            "sequential_paper_bound_010", "sequential_paper_bound_020",
            "sequential_paper_matched",
        ),
        "ablation_methods": (
            "sequential_paper_bound_010", "sequential_paper_bound_020",
        ),
        "reference_methods": ("sequential_paper_matched",),
        "cases": "distorted",
    },
    {
        "name": "oracle_k1",
        "results": Path("results/ablation_oracle"),
        "reference": Path("results/sequential"),
        "methods": ("oracle_k1_paper", "sequential_paper_matched"),
        "ablation_methods": ("oracle_k1_paper",),
        "reference_methods": ("sequential_paper_matched",),
        "cases": "distorted",
    },
    {
        "name": "joint_oracle",
        "results": Path("results/ablation_joint_oracle"),
        "reference": Path("results/distortcorrect"),
        "methods": ("dcs_paper_joint_oracle", "dcs_paper_style"),
        "ablation_methods": ("dcs_paper_joint_oracle",),
        "reference_methods": ("dcs_paper_style",),
        "cases": "distorted",
    },
)


K1_MAGNITUDE_METHODS = (
    "dcs_paper_style",
    "sequential_paper_matched",
    "sequential_paper_no_prestitch",
)


def subset_manifest(
    manifest: dict[str, Any],
    case_mode: str,
) -> dict[str, Any]:
    """Return the manifest subset required by one ablation specification."""
    if case_mode == "all":
        return manifest
    if case_mode == "distorted":
        cases = [c for c in manifest["cases"]
                 if abs(c["k1_true"]) > 1e-12]
    else:
        raise ValueError(f"unknown case mode: {case_mode}")
    subset = dict(manifest)
    subset["cases"] = cases
    subset["case_count"] = len(cases)
    return subset


def load_spec(
    spec: dict[str, Any],
    manifest: dict[str, Any],
) -> list[Record]:
    """Load an ablation's records and its matching reference arms."""
    sub_manifest = subset_manifest(manifest, spec["cases"])
    wanted_cases = {case["case"] for case in sub_manifest["cases"]}
    # Load against the full manifest first: a reference directory contains
    # both all-case and distorted-only cases, while the selected ablation may
    # intentionally cover only the latter.
    ablation_rows = [
        row for row in load_all(spec["results"], manifest)
        if row["case"] in wanted_cases
        and row["method"] in spec["ablation_methods"]
    ]
    reference_paths = spec["reference"]
    if isinstance(reference_paths, Path):
        reference_paths = (reference_paths,)
    reference_rows = [
        row for reference_path in reference_paths
        for row in load_all(reference_path, manifest)
        if row["case"] in wanted_cases
        and row["method"] in spec["reference_methods"]
    ]
    rows = ablation_rows + reference_rows
    check_complete(rows, sub_manifest, spec["methods"])
    check_results(rows)
    return rows


def load_magnitude(
    results_dir: Path,
    manifest: dict[str, Any],
) -> list[Record]:
    """Load and validate the optional lower-magnitude experiment."""
    rows = load_all(results_dir, manifest)
    check_complete(rows, manifest, K1_MAGNITUDE_METHODS)
    check_results(rows)
    return rows


def column_value(
    record: Record,
    column: str,
    ablation_name: str | None = None,
) -> Any:
    """Convert one result record field to an exportable cell value."""
    if column == "ablation":
        return ablation_name
    if column == "mi_quantization":
        return json.dumps(record.get(column), sort_keys=True)
    return record.get(column)


def row_values(
    record: Record,
    columns: Sequence[str],
    ablation_name: str | None = None,
) -> list[Any]:
    """Return a record in the requested export column order."""
    return [column_value(record, column, ablation_name) for column in columns]


def excel_records(records: list[Record]) -> list[Record]:
    """Apply reader-friendly method labels to workbook records."""
    return [
        {**record, "method": EXCEL_METHOD_NAMES.get(
            record.get("method"), record.get("method")
        )}
        for record in records
    ]


def write_csv(
    path: Path,
    columns: Sequence[str],
    records: list[Record],
    ablation_name: str | None = None,
) -> None:
    """Write a compact result table as UTF-8 CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(
            row_values(record, columns, ablation_name)
            for record in records
        )


def excel_column(index: int) -> str:
    """Convert a one-based column index to an Excel column label."""
    result = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        result = ascii_uppercase[remainder] + result
    return result


def excel_cell(reference: str, value: Any, style: int = 0) -> str:
    """Serialize one value as an inline-string or numeric XLSX cell."""
    style_attr = f' s="{style}"' if style else ""
    if value is None or value == "":
        return f'<c r="{reference}"{style_attr}/>'
    if isinstance(value, bool):
        return f'<c r="{reference}" t="b"{style_attr}><v>{int(value)}</v></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            return f'<c r="{reference}"{style_attr}/>'
        return f'<c r="{reference}"{style_attr}><v>{value}</v></c>'
    text = escape(str(value))
    return (
        f'<c r="{reference}" t="inlineStr"{style_attr}>'
        f'<is><t xml:space="preserve">{text}</t></is></c>'
    )


def worksheet_xml(
    columns: Sequence[str],
    records: list[Record],
) -> str:
    """Build one frozen-header, filtered worksheet XML document."""
    rows = [columns] + [
        row_values(record, columns)
        for record in records
    ]
    xml_rows = []
    for row_number, row in enumerate(rows, start=1):
        cells = []
        for column_number, value in enumerate(row, start=1):
            reference = f"{excel_column(column_number)}{row_number}"
            cells.append(excel_cell(reference, value, style=1 if row_number == 1 else 0))
        xml_rows.append(f'<row r="{row_number}">' + "".join(cells) + "</row>")

    widths = []
    for column_number, column in enumerate(columns, start=1):
        width = min(max(len(column) + 2, 12), 28)
        widths.append(
            f'<col min="{column_number}" max="{column_number}" '
            f'width="{width}" customWidth="1"/>'
        )
    last_cell = f"{excel_column(len(columns))}{len(rows)}"
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheetViews><sheetView workbookViewId="0">'
        '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
        '</sheetView></sheetViews>'
        '<sheetFormatPr defaultRowHeight="15"/>'
        '<cols>' + "".join(widths) + '</cols>'
        '<sheetData>' + "".join(xml_rows) + '</sheetData>'
        f'<autoFilter ref="A1:{last_cell}"/>'
        '</worksheet>'
    )


def styles_xml() -> str:
    """Return the minimal workbook style sheet used by the exporter."""
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<numFmts count="0"/>'
        '<fonts count="2">'
        '<font><sz val="11"/><name val="Calibri"/></font>'
        '<font><b/><sz val="11"/><name val="Calibri"/></font>'
        '</fonts>'
        '<fills count="3">'
        '<fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="D9EAF7"/>'
        '<bgColor indexed="64"/></patternFill></fill>'
        '</fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/>'
        '</border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" '
        'borderId="0"/></cellStyleXfs>'
        '<cellXfs count="2">'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="0"/>'
        '<xf numFmtId="0" fontId="1" fillId="2" borderId="0" '
        'applyFont="1" applyFill="1"/>'
        '</cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" '
        'builtinId="0"/></cellStyles>'
        '</styleSheet>'
    )


def write_xlsx(
    path: Path,
    sheets: list[tuple[str, list[Record]]],
    columns: Sequence[str],
) -> None:
    """Write the supplied worksheets into a dependency-free XLSX archive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
    ]
    for index in range(1, len(sheets) + 1):
        content_types.append(
            f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    content_types.append('</Types>')

    workbook_sheets = []
    workbook_rels = []
    for index, (name, _) in enumerate(sheets, start=1):
        workbook_sheets.append(
            f'<sheet name="{escape(name[:31])}" sheetId="{index}" r:id="rId{index}"/>'
        )
        workbook_rels.append(
            f'<Relationship Id="rId{index}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{index}.xml"/>'
        )
    styles_rel_id = len(sheets) + 1
    workbook_rels.append(
        f'<Relationship Id="rId{styles_rel_id}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
    )

    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<bookViews><workbookView/></bookViews><sheets>'
        + "".join(workbook_sheets)
        + '</sheets></workbook>'
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/></Relationships>'
    )
    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(workbook_rels)
        + '</Relationships>'
    )

    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "".join(content_types))
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        archive.writestr("xl/styles.xml", styles_xml())
        for index, (_, records) in enumerate(sheets, start=1):
            archive.writestr(
                f"xl/worksheets/sheet{index}.xml",
                worksheet_xml(columns, records),
            )


def main() -> None:
    """Validate ablations and write detailed and reader-friendly exports."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", type=Path, default=Path("benchmark"))
    parser.add_argument(
        "--out", type=Path,
        default=Path("results/ablation/results_summary.csv"),
        help="clean combined detailed CSV",
    )
    parser.add_argument(
        "--csv-dir", type=Path,
        default=Path("results/ablation"),
        help="directory for compact per-ablation CSVs",
    )
    parser.add_argument(
        "--xlsx", type=Path,
        default=Path("results/ablation/ablation_results.xlsx"),
        help="multi-sheet Excel workbook",
    )
    parser.add_argument(
        "--k1-magnitude-dir", type=Path, default=None,
        help="optional results directory for the separate k1-magnitude run",
    )
    parser.add_argument(
        "--k1-magnitude-bench", type=Path, default=None,
        help="manifest directory for --k1-magnitude-dir; inferred from its parent when omitted",
    )
    parser.add_argument("--no-xlsx", action="store_true")
    args = parser.parse_args()

    manifest = json.loads((args.bench / "manifest.json").read_text("utf-8"))
    all_rows = []
    grouped_rows = {}
    for spec in ABLATION_SPECS:
        rows = load_spec(spec, manifest)
        grouped_rows[spec["name"]] = rows
        all_rows.extend((spec["name"], row) for row in rows)

    magnitude_rows = []
    magnitude_dir = args.k1_magnitude_dir
    if magnitude_dir is None:
        local_magnitude_dir = Path("results/ablation_k1_magnitude")
        if local_magnitude_dir.is_dir():
            magnitude_dir = local_magnitude_dir
    if magnitude_dir is not None:
        magnitude_bench = args.k1_magnitude_bench
        if magnitude_bench is None:
            local_magnitude_bench = Path("benchmark_k1_magnitude")
            magnitude_bench = (
                local_magnitude_bench
                if local_magnitude_bench.is_dir()
                else magnitude_dir.parent / "benchmark_k1_magnitude"
            )
        magnitude_manifest = json.loads(
            (magnitude_bench / "manifest.json").read_text("utf-8")
        )
        magnitude_rows = load_magnitude(magnitude_dir, magnitude_manifest)
        all_rows.extend(("k1_magnitude", row) for row in magnitude_rows)

    order = {spec["name"]: index
             for index, spec in enumerate(ABLATION_SPECS)}
    if magnitude_rows:
        order["k1_magnitude"] = len(order)
    all_rows.sort(key=lambda item: (
        order[item[0]], item[1]["case"], item[1]["method"]
    ))
    combined_records = [row for _, row in all_rows]
    combined_names = [name for name, _ in all_rows]

    # The large CSV keeps the ablation label and all scientific measurements,
    # but drops protocol/MI/warp implementation metadata.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(
            row_values(record, CSV_COLUMNS, name)
            for name, record in zip(combined_names, combined_records)
        )

    sheets = []
    for spec in ABLATION_SPECS:
        name = spec["name"]
        records = sorted(
            grouped_rows[name],
            key=lambda row: (row["case"], row["method"]),
        )
        write_csv(args.csv_dir / f"{name}.csv", READABLE_COLUMNS, records)
        sheets.append((EXCEL_SHEET_NAMES.get(name, name), excel_records(records)))

    if magnitude_rows:
        records = sorted(
            magnitude_rows,
            key=lambda row: (row["case"], row["method"]),
        )
        write_csv(args.csv_dir / "k1_magnitude.csv", READABLE_COLUMNS, records)
        sheets.append((EXCEL_SHEET_NAMES["k1_magnitude"], excel_records(records)))

    if not args.no_xlsx:
        write_xlsx(args.xlsx, sheets, READABLE_COLUMNS)

    print(f"wrote {len(all_rows)} rows to {args.out}")
    for spec in ABLATION_SPECS:
        name = spec["name"]
        print(f"  {name}: {len(grouped_rows[name])} rows; "
              f"{args.csv_dir / (name + '.csv')}")
    if magnitude_rows:
        print(f"  k1_magnitude: {len(magnitude_rows)} rows; "
              f"{args.csv_dir / 'k1_magnitude.csv'}")
    if not args.no_xlsx:
        print(f"wrote Excel workbook to {args.xlsx}")


if __name__ == "__main__":
    main()
