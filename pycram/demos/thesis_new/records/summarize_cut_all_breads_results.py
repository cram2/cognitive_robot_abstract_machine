import argparse
import os

from demos.thesis_new.src.utils import (
    BASE_AGGREGATED_FIELDS,
    BASE_SUMMARY_FIELDS,
    append_csv,
    build_aggregated,
    build_trial_summary,
    load_rows,
)


def _bread_extra_summary_fields():
    return ["bread_name"]


def _bread_extra_summary_row(row):
    return {"bread_name": row.get("bread_name")}


def main():
    records_dir = os.path.join(os.path.dirname(__file__), "")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=os.path.join(records_dir, "cut_all_breads_results.csv"),
    )
    parser.add_argument(
        "--summary-out",
        default=os.path.join(records_dir, "cut_all_breads_results_summary.csv"),
    )
    parser.add_argument(
        "--aggregated-out",
        default=os.path.join(records_dir, "cut_all_breads_results_aggregated.csv"),
    )
    args = parser.parse_args()

    rows = load_rows(args.input)
    summary = build_trial_summary(rows, extra_row_fn=_bread_extra_summary_row)
    aggregated = build_aggregated(rows)

    summary_fields = (
        BASE_SUMMARY_FIELDS[:7]
        + _bread_extra_summary_fields()
        + BASE_SUMMARY_FIELDS[7:]
    )
    aggregated_fields = BASE_AGGREGATED_FIELDS

    append_csv(args.summary_out, summary_fields, summary)
    append_csv(args.aggregated_out, aggregated_fields, aggregated)


if __name__ == "__main__":
    main()
