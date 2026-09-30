import sys
from app.database import SessionLocal
from app.models.category import Category
from app.models.subcategory import Subcategory

TAXONOMY_DATA = [
    {
        "category": "Roads & Footpaths",
        "description": "Complaints regarding roads, pavements, potholes, and road drainage.",
        "subcategories": [
            {
                "name": "Potholes & Road Damage",
                "department": "Roads Department",
            },
            {
                "name": "Road Repair / Construction",
                "department": "Roads Department",
            },
            {
                "name": "Footpath Damage / Repair",
                "department": "Roads Department",
            },
            {
                "name": "Road Waterlogging",
                "department": "Drainage Department",
            },
        ],
    },
    {
        "category": "Waste & Sanitation",
        "description": "Complaints regarding waste collection, garbage dumping, and street cleanliness.",
        "subcategories": [
            {
                "name": "Garbage Collection / Removal",
                "department": "Waste Management Department",
            },
            {
                "name": "Garbage Dumping",
                "department": "Waste Management Department",
            },
            {
                "name": "Street / Public Area Cleaning",
                "department": "Sanitation Department",
            },
            {
                "name": "Waste Disposal & Segregation",
                "department": "Waste Management Department",
            },
        ],
    },
    {
        "category": "Water Supply",
        "description": "Complaints regarding potable water distribution, pipelines, and supply disruptions.",
        "subcategories": [
            {
                "name": "Water Supply Interruption",
                "department": "Water Supply Department",
            },
            {
                "name": "Water Leakage",
                "department": "Water Supply Department",
            },
            {
                "name": "Water Pipeline Damage / Maintenance",
                "department": "Water Supply Department",
            },
            {
                "name": "New Water Connection / Pipeline",
                "department": "Water Supply Department",
            },
        ],
    },
    {
        "category": "Sewage & Drainage",
        "description": "Complaints regarding underground sewers, manholes, overflow, and storm drains.",
        "subcategories": [
            {
                "name": "Sewer Blockage / Overflow",
                "department": "Sewerage Department",
            },
            {
                "name": "Sewer Line Repair / Maintenance",
                "department": "Sewerage Department",
            },
            {
                "name": "Drainage Blockage / Desilting",
                "department": "Drainage Department",
            },
            {
                "name": "Storm Water Drainage / Waterlogging",
                "department": "Drainage Department",
            },
        ],
    },
    {
        "category": "Street Lighting & Electricity",
        "description": "Complaints regarding public lighting, transformers, and municipal power supplies.",
        "subcategories": [
            {
                "name": "Street Light Not Working / Repair",
                "department": "Street Lighting Department",
            },
            {
                "name": "Street Light Installation",
                "department": "Street Lighting Department",
            },
            {
                "name": "Electrical Pole / Transformer Issues",
                "department": "Electrical Department",
            },
            {
                "name": "Electricity Supply",
                "department": "Electrical Department",
            },
        ],
    },
    {
        "category": "Pollution",
        "description": "Complaints regarding air, noise, water, and industrial or environmental pollution.",
        "subcategories": [
            {
                "name": "Air Pollution",
                "department": "Pollution Control Department",
            },
            {
                "name": "Noise Pollution",
                "department": "Pollution Control Department",
            },
            {
                "name": "Water Pollution",
                "department": "Pollution Control Department",
            },
            {
                "name": "Land / Waste Pollution",
                "department": "Pollution Control Department",
            },
        ],
    },
]


def seed_taxonomy():
    db = SessionLocal()
    categories_inserted = 0
    categories_skipped = 0
    subcategories_inserted = 0
    subcategories_skipped = 0

    try:
        print("Starting taxonomy seeding...")
        for item in TAXONOMY_DATA:
            cat_name = item["category"]
            cat_desc = item["description"]

            # Duplicate protection: check if category already exists
            category = db.query(Category).filter(Category.name == cat_name).first()
            if not category:
                category = Category(
                    name=cat_name,
                    description=cat_desc,
                    is_active=True,
                )
                db.add(category)
                db.flush()  # Flush to generate category.id
                categories_inserted += 1
                print(f" [+] Inserted Category: '{cat_name}'")
            else:
                categories_skipped += 1
                print(f" [~] Existing Category found: '{cat_name}'")

            for sub in item["subcategories"]:
                sub_name = sub["name"]
                dept_name = sub["department"]

                # Duplicate protection: check if subcategory exists under this category
                existing_sub = (
                    db.query(Subcategory)
                    .filter(
                        Subcategory.category_id == category.id,
                        Subcategory.name == sub_name,
                    )
                    .first()
                )

                if not existing_sub:
                    new_sub = Subcategory(
                        category_id=category.id,
                        name=sub_name,
                        department=dept_name,
                        is_active=True,
                    )
                    db.add(new_sub)
                    subcategories_inserted += 1
                    print(f"     [+] Inserted Subcategory: '{sub_name}' -> '{dept_name}'")
                else:
                    subcategories_skipped += 1
                    print(f"     [~] Existing Subcategory found: '{sub_name}'")

        db.commit()
        print("\n--- Seeding Summary ---")
        print(f"Categories:    {categories_inserted} inserted, {categories_skipped} skipped")
        print(f"Subcategories: {subcategories_inserted} inserted, {subcategories_skipped} skipped")
        print("Taxonomy seeding completed successfully!")

    except Exception as exc:
        db.rollback()
        print(f"\n[ERROR] Seeding failed: {exc}", file=sys.stderr)
        raise exc
    finally:
        db.close()


if __name__ == "__main__":
    seed_taxonomy()
