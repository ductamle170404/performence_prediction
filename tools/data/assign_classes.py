"""Create persistent student-class and class-instructor tables from OULAD registration data.

Run from the repository root:
    python tools\\data\\assign_classes.py
    python tools\\data\\assign_classes.py --data-dir data_using --class-size 40

Students are ordered by registration date within each module/presentation.
Missing dates go last; ties are broken by student ID for reproducibility.
Instructor IDs are synthetic and assigned to cover the requested teaching patterns.
"""
import argparse
from pathlib import Path

import pandas as pd


KEYS = ["code_module", "code_presentation", "id_student"]
ASSIGNMENTS_FILE = "student_class_assignments.csv"
INSTRUCTORS_FILE = "class_instructors.csv"
INSTRUCTOR_LIST_FILE = "instructors.csv"


def build_assignments(registrations: pd.DataFrame, class_size: int) -> pd.DataFrame:
    missing_columns = set(KEYS + ["date_registration"]) - set(registrations.columns)
    if missing_columns:
        raise ValueError(
            "studentRegistration.csv is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    if registrations[KEYS].isna().any().any():
        raise ValueError("Registration keys must not contain missing values.")
    if registrations.duplicated(KEYS).any():
        raise ValueError("Registration table contains duplicate student-course keys.")

    raw_dates = registrations["date_registration"]
    dates = pd.to_numeric(raw_dates, errors="coerce")
    invalid_dates = raw_dates.notna() & dates.isna()
    if invalid_dates.any():
        examples = raw_dates[invalid_dates].astype(str).drop_duplicates().head(5).tolist()
        raise ValueError(f"Unrecognized date_registration values: {examples}")

    students = registrations[KEYS].copy()
    students["date_registration"] = dates
    students["_student_id_numeric"] = pd.to_numeric(
        students["id_student"], errors="coerce"
    )
    students = students.sort_values(
        [
            "code_module",
            "code_presentation",
            "date_registration",
            "_student_id_numeric",
            "id_student",
        ],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)

    group_keys = ["code_module", "code_presentation"]
    students["registration_rank"] = students.groupby(group_keys, sort=False).cumcount() + 1
    students["class_number"] = (
        (students["registration_rank"] - 1) // class_size + 1
    )
    students["class_id"] = students.apply(
        lambda row: (
            f"{row['code_module']}-{row['code_presentation']}-"
            f"{row['class_number']:03d}"
        ),
        axis=1,
    )

    return students[
        KEYS + ["date_registration", "registration_rank", "class_id"]
    ]


def build_instructors(assignments: pd.DataFrame) -> pd.DataFrame:
    """Create reproducible synthetic instructor assignments for each class."""
    required = ["code_module", "code_presentation", "class_id"]
    missing_columns = set(required) - set(assignments.columns)
    if missing_columns:
        raise ValueError(
            "Class assignments are missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    classes = (
        assignments[required]
        .drop_duplicates()
        .sort_values(required, kind="stable")
        .reset_index(drop=True)
    )
    modules = sorted(classes["code_module"].dropna().unique().tolist())
    if len(modules) < 2:
        raise ValueError("At least two modules are required for multi-subject instructors.")

    module_rows = {
        module: classes.index[classes["code_module"].eq(module)].tolist()
        for module in modules
    }
    positions = {module: 0 for module in modules}
    instructor_ids = pd.Series(index=classes.index, dtype="string")
    next_id = 1

    def assign(rows: list[int]) -> None:
        nonlocal next_id
        instructor_ids.loc[rows] = f"SIM-GV-{next_id:04d}"
        next_id += 1

    # Pair subjects so these instructors teach multiple subjects and multiple classes.
    module_pairs = [
        (modules[index], modules[index + 1])
        for index in range(0, len(modules) - 1, 2)
    ]
    if len(modules) % 2:
        module_pairs.append((modules[-1], modules[0]))

    for first_module, second_module in module_pairs:
        first_rows = module_rows[first_module]
        second_rows = module_rows[second_module]
        first_start = positions[first_module]
        second_start = positions[second_module]
        if len(first_rows) - first_start < 2 or len(second_rows) - second_start < 2:
            raise ValueError(
                "Each paired module needs at least two unassigned classes to create "
                "multi-subject instructors."
            )
        assign(
            first_rows[first_start : first_start + 2]
            + second_rows[second_start : second_start + 2]
        )
        positions[first_module] += 2
        positions[second_module] += 2

    # Keep some one-class instructors; group the remaining same-subject classes.
    for module in modules:
        rows = module_rows[module]
        remaining = rows[positions[module] :]
        for row in remaining[:2]:
            assign([row])
        remaining = remaining[2:]
        for start in range(0, len(remaining), 3):
            assign(remaining[start : start + 3])

    if instructor_ids.isna().any():
        raise ValueError("Failed to assign an instructor to every class.")

    classes["instructor_id"] = instructor_ids
    return classes


def build_instructor_list(class_instructors: pd.DataFrame) -> pd.DataFrame:
    """Create one instructor master row per ID."""
    if "instructor_id" not in class_instructors.columns:
        raise ValueError("Class-instructor table is missing the instructor_id column.")
    instructor_ids = class_instructors["instructor_id"]
    has_blank_ids = instructor_ids.isna().any() or (
        instructor_ids.astype("string").str.strip().eq("").any()
    )
    if has_blank_ids:
        raise ValueError("Every class must have a non-empty instructor_id.")
    return (
        instructor_ids.astype("string")
        .drop_duplicates()
        .sort_values(kind="stable")
        .rename("instructor_id")
        .to_frame()
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data_using",
        help="Folder containing studentRegistration.csv and receiving the output CSVs.",
    )
    parser.add_argument(
        "--class-size",
        type=int,
        default=40,
        help="Maximum students per class (default: 40).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing outputs, including synthetic instructor assignments and roster.",
    )
    args = parser.parse_args()

    if args.class_size < 1:
        parser.error("--class-size must be at least 1")

    data_dir = args.data_dir.resolve()
    input_path = data_dir / "studentRegistration.csv"
    assignments_path = data_dir / ASSIGNMENTS_FILE
    instructors_path = data_dir / INSTRUCTORS_FILE
    instructor_list_path = data_dir / INSTRUCTOR_LIST_FILE
    if not input_path.is_file():
        parser.error(f"Input file not found: {input_path}")

    if (
        not args.overwrite
        and assignments_path.is_file()
        and instructors_path.is_file()
        and not instructor_list_path.exists()
    ):
        existing_instructors = pd.read_csv(instructors_path, dtype="string")
        instructor_list = build_instructor_list(existing_instructors)
        instructor_list.to_csv(instructor_list_path, index=False)
        print(
            f"Created {instructor_list_path} "
            f"({len(instructor_list):,} unique instructors)"
        )
        return

    existing = [
        path.name
        for path in (assignments_path, instructors_path, instructor_list_path)
        if path.exists()
    ]
    if existing and not args.overwrite:
        parser.error(
            "Output file(s) already exist: "
            + ", ".join(existing)
            + ". Use --overwrite to regenerate them (instructor assignments will be reset)."
        )

    registrations = pd.read_csv(
        input_path,
        dtype={"code_module": "string", "code_presentation": "string", "id_student": "string"},
        na_values=["?"],
    )
    assignments = build_assignments(registrations, args.class_size)
    instructors = build_instructors(assignments)
    instructor_list = build_instructor_list(instructors)

    assignments.to_csv(assignments_path, index=False, na_rep="")
    instructors.to_csv(instructors_path, index=False)
    instructor_list.to_csv(instructor_list_path, index=False)
    print(f"Created {assignments_path} ({len(assignments):,} student-course rows)")
    print(f"Created {instructors_path} ({len(instructors):,} classes)")
    print(f"Assigned {instructors['instructor_id'].nunique():,} synthetic instructors")
    print(f"Created {instructor_list_path} ({len(instructor_list):,} unique instructors)")


if __name__ == "__main__":
    main()
