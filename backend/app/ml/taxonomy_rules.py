"""Complete authoritative taxonomy rules mapping every raw (category, subcategory) pair

found in the dataset to:
- Civic status: True, False, or None (uncertain)
- Target category: One of the locked 6 categories or None
- Target subcategory: One of the locked 24 subcategories or None
- Target department: One of the locked 9 departments or None
- Mapping confidence: 'high', 'medium', 'unmapped'
- Reason: Clear audit reason
"""

# The Locked 24 Subcategories and their parent categories & departments
APPROVED_TAXONOMY = {
    ("Roads & Footpaths", "Potholes & Road Damage"): "Roads Department",
    ("Roads & Footpaths", "Road Repair / Construction"): "Roads Department",
    ("Roads & Footpaths", "Footpath Damage / Repair"): "Roads Department",
    ("Roads & Footpaths", "Road Waterlogging"): "Drainage Department",
    ("Waste & Sanitation", "Garbage Collection / Removal"): "Waste Management Department",
    ("Waste & Sanitation", "Garbage Dumping"): "Waste Management Department",
    ("Waste & Sanitation", "Street / Public Area Cleaning"): "Sanitation Department",
    ("Waste & Sanitation", "Waste Disposal & Segregation"): "Waste Management Department",
    ("Water Supply", "Water Supply Interruption"): "Water Supply Department",
    ("Water Supply", "Water Leakage"): "Water Supply Department",
    ("Water Supply", "Water Pipeline Damage / Maintenance"): "Water Supply Department",
    ("Water Supply", "New Water Connection / Pipeline"): "Water Supply Department",
    ("Sewage & Drainage", "Sewer Blockage / Overflow"): "Sewerage Department",
    ("Sewage & Drainage", "Sewer Line Repair / Maintenance"): "Sewerage Department",
    ("Sewage & Drainage", "Drain Blockage / Desilting"): "Drainage Department",
    ("Sewage & Drainage", "Storm Water Drainage / Waterlogging"): "Drainage Department",
    ("Street Lighting & Electricity", "Street Light Not Working / Repair"): "Street Lighting Department",
    ("Street Lighting & Electricity", "Street Light Installation"): "Street Lighting Department",
    ("Street Lighting & Electricity", "Electrical Pole / Transformer Issues"): "Electrical Department",
    ("Street Lighting & Electricity", "Electricity Supply"): "Electrical Department",
    ("Pollution", "Air Pollution"): "Pollution Control Department",
    ("Pollution", "Noise Pollution"): "Pollution Control Department",
    ("Pollution", "Water Pollution"): "Pollution Control Department",
    ("Pollution", "Land / Waste Pollution"): "Pollution Control Department",
}


def normalize_key(category_raw: str, subcategory_raw: str):
    c = (category_raw or "").strip().lower()
    s = (subcategory_raw or "").strip().lower()
    if c in ("nan", "null", "none"):
        c = ""
    if s in ("nan", "null", "none"):
        s = ""
    return c, s


def get_rule_mapping():
    rules = {}

    def add(cat, sub, is_civic, t_cat, t_sub, conf, reason):
        dept = None
        if t_cat and t_sub:
            dept = APPROVED_TAXONOMY.get((t_cat, t_sub))
            assert dept is not None, f"Invalid taxonomy pair: {t_cat} -> {t_sub}"
        
        c_norm, s_norm = normalize_key(cat, sub)
        rules[(c_norm, s_norm)] = {
            "is_civic": is_civic,
            "target_category": t_cat,
            "target_subcategory": t_sub,
            "target_department": dept,
            "mapping_confidence": conf,
            "reason": reason,
        }

    # ==========================================
    # 1. ROADS & FOOTPATHS
    # ==========================================
    road_cats = [
        "mobility - roads, footpaths and infrastructure",
        "roads and footpaths",
        "mobility - roads, public transport",
        "pwd",
        "",
        "null",
    ]
    for c in road_cats:
        # Potholes & Road Damage
        add(c, "fixing/reparing potholes", True, "Roads & Footpaths", "Potholes & Road Damage", "high", "Direct match for pothole repair")
        add(c, "repair of potholes on roads", True, "Roads & Footpaths", "Potholes & Road Damage", "high", "Direct match for pothole repair")
        add(c, "damaged road", True, "Roads & Footpaths", "Potholes & Road Damage", "high", "Road surface damage")
        add(c, "repair of bridge and culverts", True, "Roads & Footpaths", "Potholes & Road Damage", "high", "Bridge/culvert road damage")
        add(c, "removal of roadside debris (construction material)", True, "Roads & Footpaths", "Potholes & Road Damage", "medium", "Roadway obstruction causing road hazard")

        # Road Repair / Construction
        add(c, "tarring or asphalting of existing road", True, "Roads & Footpaths", "Road Repair / Construction", "high", "Road tarring and resurfacing")
        add(c, "tarring or asphalting of mud/kutcha/unpaved road", True, "Roads & Footpaths", "Road Repair / Construction", "high", "Unpaved road construction")
        add(c, "construction of skywalk/flyover/underpass", True, "Roads & Footpaths", "Road Repair / Construction", "high", "Road flyover/underpass civil construction")
        add(c, "construction of flyovers/ underpass", True, "Roads & Footpaths", "Road Repair / Construction", "high", "Flyover and underpass construction")
        add(c, "construction of skywalks", True, "Roads & Footpaths", "Road Repair / Construction", "high", "Skywalk civil construction")
        add(c, "need lane markings/street name/road signages", True, "Roads & Footpaths", "Road Repair / Construction", "high", "Road signage and surface markings")
        add(c, "beautification/maintenance of medians or circles", True, "Roads & Footpaths", "Road Repair / Construction", "medium", "Road traffic circle and median maintenance")

        # Footpath Damage / Repair
        add(c, "report a broken footpath", True, "Roads & Footpaths", "Footpath Damage / Repair", "high", "Footpath damage repair")
        add(c, "repair of existing footpaths", True, "Roads & Footpaths", "Footpath Damage / Repair", "high", "Footpath maintenance")
        add(c, "require a new footpath", True, "Roads & Footpaths", "Footpath Damage / Repair", "high", "Footpath construction requirement")
        add(c, "construction of new footpaths", True, "Roads & Footpaths", "Footpath Damage / Repair", "high", "Footpath construction")
        add(c, "build new footpaths and repair broken footpaths", True, "Roads & Footpaths", "Footpath Damage / Repair", "high", "Footpath construction & repair")
        add(c, "2b- repair broken footpath", True, "Roads & Footpaths", "Footpath Damage / Repair", "high", "Coded label for footpath repair")
        add(c, "2a- require new footpath", True, "Roads & Footpaths", "Footpath Damage / Repair", "high", "Coded label for footpath requirement")

        # Road Waterlogging
        add(c, "flooding/waterlogging of roads and footpaths", True, "Roads & Footpaths", "Road Waterlogging", "high", "Road waterlogging/flooding")

    # Yellow Spot category footpath anomalies
    add("yellow spot", "report a broken footpath", True, "Roads & Footpaths", "Footpath Damage / Repair", "high", "Misclassified footpath repair under yellow spot")
    add("yellow spot", "2b- repair broken footpath", True, "Roads & Footpaths", "Footpath Damage / Repair", "high", "Coded footpath repair under yellow spot")
    add("yellow spot", "require a new footpath", True, "Roads & Footpaths", "Footpath Damage / Repair", "high", "Footpath construction under yellow spot")
    add("yellow spot", "2a- require new footpath", True, "Roads & Footpaths", "Footpath Damage / Repair", "high", "Coded footpath construction under yellow spot")

    # ==========================================
    # 2. WASTE & SANITATION
    # ==========================================
    waste_cats = [
        "garbage and unsanitary practices",
        "sanitation",
        "solid waste management",
        "waste management",
    ]
    for c in waste_cats:
        # Garbage Collection / Removal
        add(c, "collection of door-to-door garbage", True, "Waste & Sanitation", "Garbage Collection / Removal", "high", "Door to door waste collection")
        add(c, "wet waste vehicle not arrived", True, "Waste & Sanitation", "Garbage Collection / Removal", "high", "Garbage collection vehicle schedule")
        add(c, "garbage vehicle not arrived", True, "Waste & Sanitation", "Garbage Collection / Removal", "high", "Garbage vehicle arrival")
        add(c, "build dry waste collection centre", True, "Waste & Sanitation", "Garbage Collection / Removal", "high", "Dry waste collection infrastructure")
        add(c, "maintenance of dry waste collection centre", True, "Waste & Sanitation", "Garbage Collection / Removal", "high", "Dry waste collection center maintenance")

        # Garbage Dumping
        add(c, "clearance of garbage dump or black spot", True, "Waste & Sanitation", "Garbage Dumping", "high", "Garbage black spot clearance")
        add(c, "garbage dumping in vacant lot/land", True, "Waste & Sanitation", "Garbage Dumping", "high", "Illegal waste dumping")
        add(c, "garbage dump", True, "Waste & Sanitation", "Garbage Dumping", "high", "Garbage dump clearance")
        add(c, "clearing of roadside dustbin", True, "Waste & Sanitation", "Garbage Dumping", "high", "Roadside dustbin dump clearance")
        add(c, "dustbins not cleared", True, "Waste & Sanitation", "Garbage Dumping", "high", "Uncleared dustbins")
        add(c, "debris removal or construction material", True, "Waste & Sanitation", "Garbage Dumping", "high", "Debris and waste dump removal")

        # Street / Public Area Cleaning
        add(c, "regular sweeping of streets", True, "Waste & Sanitation", "Street / Public Area Cleaning", "high", "Street sanitation sweeping")
        add(c, "sweeping not done", True, "Waste & Sanitation", "Street / Public Area Cleaning", "high", "Street sweeping grievance")
        add(c, "cleaning of litter bins", True, "Waste & Sanitation", "Street / Public Area Cleaning", "high", "Public litter bin cleaning")
        add(c, "removal of dead animals", True, "Waste & Sanitation", "Street / Public Area Cleaning", "high", "Dead animal clearance for public health")
        add(c, "fogging (mosquito menace) or pest control", True, "Waste & Sanitation", "Street / Public Area Cleaning", "medium", "Vector and pest control public sanitation")
        add(c, "improper disposal of feacal waste or  septage", True, "Waste & Sanitation", "Street / Public Area Cleaning", "high", "Fecal waste dumping public sanitation")

        # Waste Disposal & Segregation
        add(c, "stop/prevent burning of garbage", True, "Waste & Sanitation", "Waste Disposal & Segregation", "high", "Waste burning prevention")
        add(c, "implementation of waste segregation/garbage", True, "Waste & Sanitation", "Waste Disposal & Segregation", "high", "Waste segregation implementation")
        add(c, "build garbage/waste composting units", True, "Waste & Sanitation", "Waste Disposal & Segregation", "high", "Waste processing & composting facility")

    # Yellow spot and public toilets cleaning
    add("yellow spot", "report a public urination or yellow spot", True, "Waste & Sanitation", "Street / Public Area Cleaning", "high", "Public area urination cleaning")
    add("yellow spot", "1a- address public urination or yellow spots", True, "Waste & Sanitation", "Street / Public Area Cleaning", "high", "Coded public area urination cleaning")
    add("yellow spot", "report garbage or debris on footpath", True, "Waste & Sanitation", "Street / Public Area Cleaning", "high", "Footpath debris cleaning")
    add("yellow spot", "2c- remove garbage or debris on footpath", True, "Waste & Sanitation", "Street / Public Area Cleaning", "high", "Coded footpath debris cleaning")
    add("yellow spot", "report a dirty or unusable public toilet", True, "Waste & Sanitation", "Street / Public Area Cleaning", "high", "Public toilet cleaning")
    add("yellow spot", "report a dirty or unusable mobile public toilet", True, "Waste & Sanitation", "Street / Public Area Cleaning", "high", "Mobile toilet cleaning")
    add("yellow spot", "require a new public toilet", True, "Waste & Sanitation", "Street / Public Area Cleaning", "medium", "Public sanitation toilet infrastructure")
    add("yellow spot", "1b- require new public toilet", True, "Waste & Sanitation", "Street / Public Area Cleaning", "medium", "Public sanitation toilet infrastructure")
    add("yellow spot", "1c- repair dirty or unusable public toilet", True, "Waste & Sanitation", "Street / Public Area Cleaning", "high", "Toilet cleaning & repair")
    add("public toilets", "cleaning of public toilet", True, "Waste & Sanitation", "Street / Public Area Cleaning", "high", "Public toilet cleaning")
    add("public toilets", "1c- repair dirty or unusable public toilet", True, "Waste & Sanitation", "Street / Public Area Cleaning", "high", "Toilet cleaning & repair")
    add("public toilets", "construction of new public toilet", True, "Waste & Sanitation", "Street / Public Area Cleaning", "medium", "Public toilet sanitation infrastructure")
    add("public toilets", "1b- require new public toilet", True, "Waste & Sanitation", "Street / Public Area Cleaning", "medium", "Public toilet requirement")
    add("public toilets", "urination in public or open defecation", True, "Waste & Sanitation", "Street / Public Area Cleaning", "high", "Open defecation / urination public sanitation")
    add("public toilets", "maintenance/repair of toilet fixtures", True, "Waste & Sanitation", "Street / Public Area Cleaning", "high", "Toilet fixture maintenance")
    add("public toilets", "cleaning of septic tank of public toilets", True, "Waste & Sanitation", "Street / Public Area Cleaning", "high", "Toilet septic tank cleaning")

    # Parks public toilet anomalies
    add("parks & recreation", "construction of new public toilet in parks", True, "Waste & Sanitation", "Street / Public Area Cleaning", "medium", "Public toilet in park")
    add("parks & recreation", "maintenance of existing public toilet in parks", True, "Waste & Sanitation", "Street / Public Area Cleaning", "medium", "Public toilet in park maintenance")
    add("playgrounds", "construction of new public toilet in playground", True, "Waste & Sanitation", "Street / Public Area Cleaning", "medium", "Public toilet in playground")

    # Null category public sanitation anomalies
    add("null", "report a public urination or yellow spot", True, "Waste & Sanitation", "Street / Public Area Cleaning", "high", "Public urination cleaning")
    add("null", "report a dirty or unusable public toilet", True, "Waste & Sanitation", "Street / Public Area Cleaning", "high", "Toilet cleaning")
    add("null", "report garbage or debris on footpath", True, "Waste & Sanitation", "Street / Public Area Cleaning", "high", "Footpath debris cleaning")

    # Animal husbandry dead animal cleaning
    add("animal husbandry", "removal of dead animals", True, "Waste & Sanitation", "Street / Public Area Cleaning", "high", "Dead animal clearance")
    add("covid 19", "request for fogging/sanitation during covid-19", True, "Waste & Sanitation", "Street / Public Area Cleaning", "medium", "Sanitation fogging during pandemic")
    add("fire pollution", "burning of garbage in open space", True, "Waste & Sanitation", "Waste Disposal & Segregation", "high", "Garbage burning in open space")

    # ==========================================
    # 3. WATER SUPPLY
    # ==========================================
    water_cats = ["water supply and services", "water supply"]
    for c in water_cats:
        # Water Supply Interruption
        add(c, "regular water supply", True, "Water Supply", "Water Supply Interruption", "high", "Water supply shortage/irregularity")
        add(c, "maintain regular water supply", True, "Water Supply", "Water Supply Interruption", "high", "Water supply maintenance")
        add(c, "regular supply of water", True, "Water Supply", "Water Supply Interruption", "high", "Water supply irregularity")

        # Water Leakage
        add(c, "stop water leakage", True, "Water Supply", "Water Leakage", "high", "Water pipe leakage")
        add(c, "reduce water leakage and wastage", True, "Water Supply", "Water Leakage", "high", "Water pipe wastage and leak")

        # Water Pipeline Damage / Maintenance
        add(c, "maintenance of existing water pipeline", True, "Water Supply", "Water Pipeline Damage / Maintenance", "high", "Water pipeline maintenance")
        add(c, "maintenance of existing water tank", True, "Water Supply", "Water Pipeline Damage / Maintenance", "high", "Water tank infrastructure maintenance")
        add(c, "illegal pump", True, "Water Supply", "Water Pipeline Damage / Maintenance", "high", "Illegal booster pump causing pipeline suction/damage")
        add(c, "illegal connection", True, "Water Supply", "Water Pipeline Damage / Maintenance", "high", "Illegal water pipeline tapping")

        # New Water Connection / Pipeline
        add(c, "provision for a new water pipeline", True, "Water Supply", "New Water Connection / Pipeline", "high", "New water pipeline provision")
        add(c, "water meter", True, "Water Supply", "New Water Connection / Pipeline", "high", "Water connection meter")
        add(c, "installation of drinking water atms", True, "Water Supply", "New Water Connection / Pipeline", "high", "Public drinking water facility")
        add(c, "increase public taps and borewells-water pumps", True, "Water Supply", "New Water Connection / Pipeline", "high", "Public water connection expansion")

    # ==========================================
    # 4. SEWAGE & DRAINAGE
    # ==========================================
    sewer_drain_cats = ["sewerage systems", "storm water drains", "sanitation", "solid waste management", "garbage and unsanitary practices"]
    for c in sewer_drain_cats:
        # Sewer Blockage / Overflow
        add(c, "sewarage or storm water overflow", True, "Sewage & Drainage", "Sewer Blockage / Overflow", "high", "Sewage overflow")
        add(c, "removal of sewer pipe blockages and overflows", True, "Sewage & Drainage", "Sewer Blockage / Overflow", "high", "Sewer pipe blockage clearance")
        add(c, "maintenance and repair of manholes", True, "Sewage & Drainage", "Sewer Blockage / Overflow", "high", "Manhole maintenance and repair")
        add(c, "open manholes or drains", True, "Sewage & Drainage", "Sewer Blockage / Overflow", "high", "Hazardous open manholes or drains")

        # Sewer Line Repair / Maintenance
        add(c, "maintenance and repair of sewage lines", True, "Sewage & Drainage", "Sewer Line Repair / Maintenance", "high", "Sewage line maintenance")
        add(c, "maintanence of sewage lines", True, "Sewage & Drainage", "Sewer Line Repair / Maintenance", "high", "Sewage line maintenance")
        add(c, "construction of sewage lines", True, "Sewage & Drainage", "Sewer Line Repair / Maintenance", "high", "Sewage line civil construction")
        add(c, "maintenance of sewage treatment plant", True, "Sewage & Drainage", "Sewer Line Repair / Maintenance", "medium", "STP maintenance")

        # Drain Blockage / Desilting
        add(c, "desilting/remove blockage of storm water drains", True, "Sewage & Drainage", "Drain Blockage / Desilting", "high", "Storm drain blockage desilting")
        add(c, "desilting existing roadside drains", True, "Sewage & Drainage", "Drain Blockage / Desilting", "high", "Roadside drain desilting")
        add(c, "desilting of storm water drains", True, "Sewage & Drainage", "Drain Blockage / Desilting", "high", "Storm drain desilting")
        add(c, "construction of roadside drains", True, "Sewage & Drainage", "Drain Blockage / Desilting", "high", "Roadside drain construction")
        add(c, "construction of storm water drains", True, "Sewage & Drainage", "Drain Blockage / Desilting", "high", "Storm water drain construction")
        add(c, "need covering slabs for roadside drains", True, "Sewage & Drainage", "Drain Blockage / Desilting", "high", "Roadside drain covering slabs")
        add(c, "repair  of broken storm water drains", True, "Sewage & Drainage", "Drain Blockage / Desilting", "high", "Broken storm drain repair")
        add(c, "sidewall and fencing for storm water drains", True, "Sewage & Drainage", "Drain Blockage / Desilting", "high", "Storm drain fencing & sidewall")
        add(c, "need roadside rainwater harvesting system", True, "Sewage & Drainage", "Drain Blockage / Desilting", "high", "Roadside drain rainwater harvesting")
        add(c, "repair of roadside rainwater harvesting system", True, "Sewage & Drainage", "Drain Blockage / Desilting", "high", "Rainwater drain repair")
        add(c, "cleaning of drain", True, "Sewage & Drainage", "Drain Blockage / Desilting", "high", "Drain cleaning")

        # Storm Water Drainage / Waterlogging
        add(c, "maintenance and repair of storm water drains", True, "Sewage & Drainage", "Storm Water Drainage / Waterlogging", "high", "Storm water drain system maintenance")

    # Road category drain cross-over
    add("mobility - roads, footpaths and infrastructure", "construction of roadside drains", True, "Sewage & Drainage", "Drain Blockage / Desilting", "high", "Roadside drain construction under roads category")
    add("mobility - roads, footpaths and infrastructure", "desilting existing roadside drains", True, "Sewage & Drainage", "Drain Blockage / Desilting", "high", "Roadside drain desilting under roads category")
    add("mobility - roads, footpaths and infrastructure", "need covering slabs for roadside drains", True, "Sewage & Drainage", "Drain Blockage / Desilting", "high", "Drain slabs under roads category")
    add("mobility - roads, footpaths and infrastructure", "need roadside rainwater harvesting system", True, "Sewage & Drainage", "Drain Blockage / Desilting", "high", "Rainwater drain under roads category")
    add("mobility - roads, footpaths and infrastructure", "repair of roadside rainwater harvesting system", True, "Sewage & Drainage", "Drain Blockage / Desilting", "high", "Rainwater drain repair under roads category")

    # ==========================================
    # 5. STREET LIGHTING & ELECTRICITY
    # ==========================================
    light_elec_cats = [
        "street lighting",
        "streetlights",
        "electricity and power supply",
        "electricity & power",
        "power supply",
    ]
    for c in light_elec_cats:
        # Street Light Not Working / Repair
        add(c, "maintenance/repair of streetlights", True, "Street Lighting & Electricity", "Street Light Not Working / Repair", "high", "Streetlight repair")
        add(c, "installation/maintenance of streetlight timer", True, "Street Lighting & Electricity", "Street Light Not Working / Repair", "high", "Streetlight timer maintenance")

        # Street Light Installation
        add(c, "installation of new streetlights", True, "Street Lighting & Electricity", "Street Light Installation", "high", "New streetlight installation")
        add(c, "installation of lights in playground", True, "Street Lighting & Electricity", "Street Light Installation", "high", "Public facility light installation")
        add(c, "installation of lights in parks", True, "Street Lighting & Electricity", "Street Light Installation", "high", "Public park lighting installation")
        add(c, "maintenance of lights in parks", True, "Street Lighting & Electricity", "Street Light Installation", "high", "Public park lighting")
        add(c, "maintenance of lights in playground", True, "Street Lighting & Electricity", "Street Light Installation", "high", "Public playground lighting")

        # Electrical Pole / Transformer Issues
        add(c, "repair of transformers/electric poles", True, "Street Lighting & Electricity", "Electrical Pole / Transformer Issues", "high", "Electric pole and transformer repair")
        add(c, "repair of transformers/ electric poles", True, "Street Lighting & Electricity", "Electrical Pole / Transformer Issues", "high", "Electric pole and transformer repair (with space)")
        add(c, "better maintenance of transformers", True, "Street Lighting & Electricity", "Electrical Pole / Transformer Issues", "high", "Transformer maintenance")
        add(c, "installation of new transformers/electric poles", True, "Street Lighting & Electricity", "Electrical Pole / Transformer Issues", "high", "Electric pole installation")
        add(c, "relocation of transformers/electric poles", True, "Street Lighting & Electricity", "Electrical Pole / Transformer Issues", "high", "Electric pole relocation")
        add(c, "dangling or hanging of ofcs/cables/wires", True, "Street Lighting & Electricity", "Electrical Pole / Transformer Issues", "high", "Hazardous hanging electrical/cable wires")

        # Electricity Supply
        add(c, "regular supply of electricity", True, "Street Lighting & Electricity", "Electricity Supply", "high", "Power outage and electricity supply")
        add(c, "provide 24 hour power supply", True, "Street Lighting & Electricity", "Electricity Supply", "high", "Uninterrupted power supply request")
        add(c, "ensure efficient and uninterrupted power supply", True, "Street Lighting & Electricity", "Electricity Supply", "high", "Uninterrupted power supply")

    # Cable wire under roads category
    add("mobility - roads, footpaths and infrastructure", "dangling or hanging of ofcs/cables/wires", True, "Street Lighting & Electricity", "Electrical Pole / Transformer Issues", "high", "Hanging cables over streets")
    add("playgrounds", "installation of lights in playground", True, "Street Lighting & Electricity", "Street Light Installation", "high", "Playground lights installation")
    add("playgrounds", "maintenance of lights in playground", True, "Street Lighting & Electricity", "Street Light Installation", "high", "Playground lights maintenance")
    add("parks & recreation", "installation of lights in parks", True, "Street Lighting & Electricity", "Street Light Installation", "high", "Park lights installation")
    add("parks & recreation", "maintenance of lights in parks", True, "Street Lighting & Electricity", "Street Light Installation", "high", "Park lights maintenance")

    # ==========================================
    # 6. POLLUTION
    # ==========================================
    for c in ["pollution", "fire pollution"]:
        add(c, "air pollution", True, "Pollution", "Air Pollution", "high", "Air pollution")
        add(c, "noise pollution", True, "Pollution", "Noise Pollution", "high", "Noise pollution")
        add(c, "water pollution", True, "Pollution", "Water Pollution", "high", "Water pollution")
        add(c, "land pollution", True, "Pollution", "Land / Waste Pollution", "high", "Land pollution")
        add(c, "pollution - others", True, "Pollution", "Air Pollution", "medium", "General environmental pollution")
        add(c, "fire pollution", True, "Pollution", "Air Pollution", "medium", "Smoke/fire emissions")

    for c in ["crime and safety"]:
        add(c, "playing of loudspeakers without permisson", True, "Pollution", "Noise Pollution", "high", "Loudspeaker noise pollution")

    for c in ["lakes"]:
        add(c, "water pollution", True, "Pollution", "Water Pollution", "high", "Lake water pollution")
        add(c, "maintenance of lake surrounding", True, "Pollution", "Water Pollution", "medium", "Lake waterbody protection and cleaning")
        add(c, "desilting/remove blockage in lakes", True, "Pollution", "Water Pollution", "medium", "Lake environmental desilting")
        add(c, "removal of encroachment in lakes", True, "Pollution", "Water Pollution", "medium", "Lake catchment environmental protection")
        add(c, "sidewalls and fencing for lake", True, "Pollution", "Water Pollution", "medium", "Lake environmental protection")

    # ==========================================
    # 7. EXPLICIT NON-CIVIC / OUT-OF-SCOPE
    # ==========================================
    # Traffic Violations (Police Enforcement, NOT municipal civic infrastructure)
    traffic_police_items = [
        "riding without a helmet",
        "no parking",
        "wrong parking",
        "parking on footpath",
        "traffic jams/congestion or bottlenecks",
        "defective silencer",
        "using mobile phone",
        "jumping traffic signal",
        "reckless driving",
        "pillion rider without helmet",
        "pillion rider not wearing helmet",
        "triple riding",
        "using black film-other materials",
        "violating lane discipline",
        "autorickshaws meter issues",
        "autorickshaw refusing hire",
        "over speeding",
        "tinted glass on car",
        "driving on wrong side",
        "no entry violation",
        "one way/no entry",
        "riding on footpath",
        "encroachment of cycle lane",
        "stopped on zebra cross",
        "defective/fancy number plate",
        "installation of traffic sign boards",
        "installation of traffic or pedestrain lights",
        "maintenance/repair of traffic or pedestrain lights",
        "repair of  mast light at a busy road (main road) junction",
        "repair of pedestrian lights",
        "provision for parking",
        "taking a u-turn where u-turn is prohibited",
        "stunt riding",
        "not wearing seat belt",
        "installation of traffic surveillance camera",
    ]
    for sub in traffic_police_items:
        add("traffic and road safety", sub, False, None, None, "high", "Traffic police law enforcement - outside municipal civil domain")

    # Traffic lights under other categories
    add("mobility - roads, public transport", "installation of traffic lights", False, None, None, "high", "Traffic signal enforcement equipment")

    # Animal Control / Husbandry
    animal_items = [
        "stray dog sterilisation/animal birth control (abc)",
        "controlling of stray pigs",
        "controlling of stray cattle or maintenance of cattle pound",
        "dog",
        "maintenance of city meat production and processing centre",
        "animal catcher",
        "rabies vaccination",
    ]
    for sub in animal_items:
        for c in ["animal husbandry", "animal catcher"]:
            add(c, sub, False, None, None, "high", "Animal control/veterinary services - outside municipal civil infrastructure")

    # Revenue / Documents / Certificates
    admin_docs = [
        "property tax",
        "khata certificate",
        "birth or death certificate",
        "certificates - other",
        "other certificates",
        "pension",
        "ration card",
        "ration card - others",
        "emp grievance redressal",
        "gradation list",
        "showing old owner name in online property tax system",
    ]
    for sub in admin_docs:
        for c in ["certificates", "pension", "ration card", "emp grievance redressal", "others"]:
            add(c, sub, False, None, None, "high", "Administrative / revenue / citizen identity document service")

    # Encroachment / Police / Law & Order
    law_order_items = [
        "government land/property encroachment",
        "stop unauthorised construction",
        "fencing of bbmp/government land",
        "construction of residential quarters",
        "construction of shopping and commercial complexes",
        "maintenance of shopping and commercial complexes",
        "maintenance of existing council buildings",
        "maintenance of existing ward & zonal offices",
        "maintenance of crematoriums and burial ground",
        "construction of crematoriums and burial ground",
        "construction of markets",
        "construction of stadiums",
        "maintenance of educational institutions and libraries",
        "construction of educational institutions and libraries",
        "maintenance/renovation of heritage sites",
        "removal of illegal posters and hoardings",
        "management of hawkers and vendors",
        "provision for parking",
        "installation of surveillance cameras",
        "maintenance of surveillance cameras",
        "increase number of beat patrols",
        "increase police beat patrolling",
        "introduce neighbourhood policing",
        "two wheeler wheeling",
        "eve teasing/public nuisance",
        "shops/establishments being open in late hours",
        "proibition & sale of tobacco products, plastic carry bags",
        "spitting",
        "fire safety - others",
        "fire exit blocked",
        "fire pump is on off mode",
        "violation of lockdown during covid-19",
        "violation of quarantine during covid-19",
    ]
    for sub in law_order_items:
        for c in ["community infrastructure and services", "crime and safety", "safety and crime", "covid 19", "fire safety", "mobility - roads, footpaths and infrastructure", "proibition & sale of tobacco products, plastic carry bags"]:
            add(c, sub, False, None, None, "high", "Regulatory enforcement / policing / municipal real estate administration")

    # Parks & Forestry
    park_tree_items = [
        "maintenance of existing parks",
        "construction of new parks",
        "cleanliness of public garden",
        "maintenance of existing playground",
        "construction of new playground",
        "maintenance of existing swimming pools",
        "maintenance of trees/saplings",
        "plant saplings/trees in your locality",
        "plant new tree saplings in parks",
        "defacement of trees",
        "prevent unauthorized tree cutting",
        "remove unwanted tree guards",
        "tree guards required",
        "3a- require playing and gym equipment",
    ]
    for sub in park_tree_items:
        for c in ["parks & recreation", "parks & garden", "trees and saplings", "playgrounds", "yellow spot", "community infrastructure and services"]:
            add(c, sub, False, None, None, "high", "Horticulture / urban forestry / public recreational grounds")

    # Public Transport
    transit_items = [
        "construction of new bus shelter",
        "maintenance/repair of bus shelter",
        "issue with bus driver or conductor",
        "provide new bus routes",
        "ksrtc - need new bus route",
        "increase frequency of bus services",
        "build new bus stands or shelters",
        "build or improve metro rail network",
        "better quality and maintenance of buses",
        "introduction of new bus routes",
        "ksrtc - driver or conductor",
        "ksrtc - others",
    ]
    for sub in transit_items:
        for c in ["public transport - bmtc", "public transport - ksrtc", "public transport (bmtc and metro)", "mobility - roads, public transport"]:
            add(c, sub, False, None, None, "high", "State transport bus corporation operational request")

    # ==========================================
    # 8. GENERIC / UNMAPPED / UNCERTAIN
    # ==========================================
    for c in ["others", "", "null", "pension"]:
        for s in ["others", "others - others", "", "null"]:
            add(c, s, None, None, None, "unmapped", "Completely generic label lacking complaint specificity")

    return rules
