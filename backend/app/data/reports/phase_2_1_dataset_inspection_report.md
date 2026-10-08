# Phase 2.1 — Dataset Inspection Report

Project: Municipal Complaint Intelligence System
Dataset File: backend/app/data/raw/5f99b09a-64b5-45f0-ab18-4cf0a0cabf6d.csv
Execution Context: Read-only statistical inspection (no modifications, deletions, cleaning, or ML training performed).
Date: 2026-10-08

================================================================================
1. DATASET OVERVIEW
================================================================================
- Total Records (Rows): 16,071 complaints
- Total Columns: 17 features
- File Size: 8.02 MB (8,414,147 bytes)
- Dataset Domain: Real-world civic complaints filed across Bengaluru municipal wards (BBMP jurisdiction) from 2019 to 2022.

================================================================================
2. COLUMN ANALYSIS
================================================================================
List of all 17 columns with types, missing counts, unique values, and samples:

1. created_at
   - Type: String (DateTime)
   - Missing: 0 (0.00%)
   - Unique Values: 15,229
   - Sample Values: '1-1-2019 06:33', '3/31/2019 18:18', '2-5-2019 12:26'

2. ward_id
   - Type: Integer (String representation)
   - Missing: 0 (0.00%)
   - Unique Values: 198
   - Sample Values: '22', '27', '17', '110'

3. title
   - Type: Text / String
   - Missing: 0 (0.00%)
   - Unique Values: 13,582
   - Sample Values: 'Govt Road is encroached with knowledge of officials', 'Potholes on 80ft road'

4. description
   - Type: Text / Long Narrative String
   - Missing: 0 (0.00%)
   - Unique Values: 15,485
   - Sample Values: Full citizen grievances describing issue details

5. sub_category_id
   - Type: Integer / ID
   - Missing: 18 (0.11%)
   - Unique Values: 221
   - Sample Values: '162', '42', '105', 'NULL'

6. civic_agency_id
   - Type: Integer / ID
   - Missing: 655 (4.08%)
   - Unique Values: 16
   - Sample Values: '10', '1', '3', 'NULL'

7. location
   - Type: Geocoded Address String
   - Missing: 0 (0.00%)
   - Unique Values: 12,096
   - Sample Values: '9Th Cross St, Hanumanthappa Layout, Sultanpalya, Hebbal, Bengaluru'

8. address
   - Type: User Landmark / Address String
   - Missing: 7,193 (44.76%)
   - Unique Values: 8,069
   - Sample Values: 'Opposite S M Food Palace', 'Near Metro Pillar 42', 'NULL'

9. latitude
   - Type: Float (WGS84)
   - Missing: 0 (0.00%)
   - Unique Values: 12,936
   - Sample Values: 13.0318055, 12.9715987

10. longitude
   - Type: Float (WGS84)
   - Missing: 0 (0.00%)
   - Unique Values: 12,747
   - Sample Values: 77.6054367, 77.5945627

11. ward_title
   - Type: Text / Ward Name
   - Missing: 32 (0.20%)
   - Unique Values: 199
   - Sample Values: 'Vishwanath Nagenahalli', 'Banasavadi', 'Koramangala'

12. category_id
   - Type: Integer / ID
   - Missing: 32 (0.20%)
   - Unique Values: 44
   - Sample Values: '382', '9', '29', 'NULL'

13. category_title
   - Type: Text / Category Name
   - Missing: 32 (0.20%)
   - Unique Values: 44
   - Sample Values: 'Mobility - Roads, Footpaths and Infrastructure', 'Garbage and Unsanitary Practices'

14. sub_category_title
   - Type: Text / Subcategory Name
   - Missing: 18 (0.11%)
   - Unique Values: 221
   - Sample Values: 'Fixing/Reparing Potholes', 'Clearance Of Garbage Dump Or Black Spot'

15. civic_agency_title
   - Type: Text / Agency Acronym or Name
   - Missing: 655 (4.08%)
   - Unique Values: 16
   - Sample Values: 'BBMP', 'BWSSB', 'BESCOM', 'BTP', 'KSPCB'

16. complaint_status_title
   - Type: Text / Lifecycle Status
   - Missing: 0 (0.00%)
   - Unique Values: 6
   - Sample Values: 'Open', 'Resolved', 'On-the-Job', 'Rejected', 'Re-opened', 'Closed'

17. comment_count
   - Type: Integer
   - Missing: 0 (0.00%)
   - Unique Values: 45
   - Sample Values: 0, 1, 6, 115

================================================================================
3. DUPLICATE ANALYSIS
================================================================================
- Exact Duplicate Rows (across all 17 columns): 33 rows
- Duplicate (title, description) Pairs: 586 pairs
- Duplicate Titles: 2,489 instances
- Analysis of Duplicates:
  * Repeated quick submissions from the exact same citizen at the same location (user retries/spam).
  * Independent complaints from multiple citizens reporting the same physical road pothole, garbage dump, or sewer line overflow in the same locality.

================================================================================
4. COMPLAINT TEXT ANALYSIS
================================================================================
The primary text fields are 'title' (headline) and 'description' (detailed complaint text).

Field 'title':
- Empty / Missing: 0 (0.00%)
- Short (< 10 chars): 402 (2.50%)
- Minimum Length: 3 characters
- Maximum Length: 190 characters
- Average Length: 44.17 characters (~7 words)
- Median Length: 42 characters

Field 'description':
- Empty / Missing: 0 (0.00%)
- Short (< 10 chars): 0 (0.00%)
- Minimum Length: 10 characters
- Maximum Length: 3,740 characters
- Average Length: 223.63 characters (~38 words)
- Median Length: 162 characters

Text Examples:
1. Title: "Govt Road is encroached with knowledge of officials"
   Description: "Govt Road is encroached by Private buildings in last 3 months, even BWSSB pipe has gone under their construction."
2. Title: "Sewage water overflow"
   Description: "From past one week sewage water is overflowing on the road. It is impossible to walk and bad odor is spreading all over the layout."
3. Title: "Clearance Of Garbage Dump Or Black Spot"
   Description: "Door to door garbage vehicle has not visited 4th cross for four consecutive days. Waste is piling up on the vacant plot."

================================================================================
5. CATEGORY ANALYSIS
================================================================================
- Total Unique Raw Categories: 43 distinct valid categories + 1 NULL (32 records)
- Total Distinct Values: 44

Category Distribution:
1. Mobility - Roads, Footpaths and Infrastructure : 5,072 (31.56%)
2. Garbage and Unsanitary Practices               : 3,946 (24.55%)
3. Traffic and Road Safety                        :   967  (6.02%)
4. Yellow Spot                                    :   948  (5.90%)
5. Animal Husbandry                               :   859  (5.35%)
6. Street lighting                                :   807  (5.02%)
7. Streetlights                                   :   693  (4.31%)
8. Pollution                                      :   437  (2.72%)
9. Others                                         :   346  (2.15%)
10. Water Supply and Services                     :   320  (1.99%)
11. Sewerage Systems                              :   301  (1.87%)
12. Electricity and Power Supply                  :   228  (1.42%)
13. Crime and Safety                              :   143  (0.89%)
14. Storm Water Drains                            :   138  (0.86%)
15. Community Infrastructure and Services         :   133  (0.83%)
16. Parks & Recreation                            :   121  (0.75%)
17. Roads and Footpaths                           :   113  (0.70%)
18. Trees and Saplings                            :   104  (0.65%)
19. Public Toilets                                :    61  (0.38%)
20. Lakes                                         :    54  (0.34%)
21. Certificates                                  :    53  (0.33%)
22. Sanitation                                    :    39  (0.24%)
23. Public Transport - BMTC                       :    35  (0.22%)
24. NULL (Missing)                                :    32  (0.20%)
25. Water Supply                                  :    21  (0.13%)
26. Mobility - Roads, Public transport            :    19  (0.12%)
27. Covid 19                                      :    11  (0.07%)
28. Safety and Crime                              :    11  (0.07%)
29. Fire Safety                                   :    10  (0.06%)
30. Electricity & Power                           :     8  (0.05%)
31. Playgrounds                                   :     7  (0.04%)
32. Public transport (BMTC and Metro)             :     7  (0.04%)
33. PWD                                           :     5  (0.03%)
34. Solid Waste Management                        :     5  (0.03%)
35. Power supply                                  :     5  (0.03%)
36. Public Transport - KSRTC                      :     4  (0.02%)
37. Pension                                       :     1  (0.01%)
38. EMP Grievance Redressal                       :     1  (0.01%)
39. Ration Card                                   :     1  (0.01%)
40. Fire Pollution                                :     1  (0.01%)
41. Animal Catcher                                :     1  (0.01%)
42. Prohibition & Sale of Tobacco/Plastic         :     1  (0.01%)
43. Parks & Garden                                :     1  (0.01%)
44. Waste Management                              :     1  (0.01%)

================================================================================
6. SUBCATEGORY ANALYSIS
================================================================================
- Total Raw Subcategories: 220 unique valid titles + 18 NULL records
- Total Distinct Values: 221

Top 30 Subcategories by Record Count:
 1. Clearance Of Garbage Dump Or Black Spot          : 3,110 (19.35%)
 2. Fixing/Reparing Potholes                         : 2,247 (13.98%)
 3. Tarring Or Asphalting Of Existing Road           : 1,623 (10.10%)
 4. Maintenance/Repair Of Streetlights               : 1,294  (8.05%)
 5. Stray Dog Sterilisation/Animal Birth Control     :   772  (4.80%)
 6. Collection Of Door-to-door Garbage               :   361  (2.25%)
 7. Others                                           :   323  (2.01%)
 8. Tarring Or Asphalting Of Mud/Kutcha/Unpaved Road :   260  (1.62%)
 9. Report A Public Urination or Yellow Spot         :   257  (1.60%)
10. Noise Pollution                                  :   253  (1.57%)
11. Garbage Dumping In Vacant Lot/Land               :   240  (1.49%)
12. Regular Water Supply                             :   236  (1.47%)
13. Flooding/Waterlogging Of Roads And Footpaths     :   226  (1.41%)
14. Traffic Jams/Congestion Or Bottlenecks           :   210  (1.31%)
15. Report Garbage or Debris on Footpath             :   197  (1.23%)
16. Regular Supply Of Electricity                    :   190  (1.18%)
17. Report A Broken Footpath                         :   181  (1.13%)
18. Installation Of New Streetlights                 :   158  (0.98%)
19. Riding Without A Helmet                          :   144  (0.90%)
20. Maintenance And Repair Of Sewage Lines           :   143  (0.89%)
21. No Parking                                       :   122  (0.76%)
22. Eve Teasing/Public Nuisance                      :   118  (0.73%)
23. Construction Of Roadside Drains                  :   117  (0.73%)
24. Construction of new footpaths                    :   116  (0.72%)
25. Build new footpaths and repair broken footpaths  :   113  (0.70%)
26. Air Pollution                                    :   112  (0.70%)
27. Wrong Parking                                    :   112  (0.70%)
28. Require A New Footpath                           :   103  (0.64%)
29. Parking On Footpath                              :    98  (0.61%)
30. Repair Of Existing Footpaths                     :    95  (0.59%)

Bottom 15 Subcategories (Sparse / Long-Tail):
- Cleanliness of public Garden                       : 1
- Cleaning of litter bins                            : 1
- Regular supply of water                            : 1
- Installation Of Lights In Playground               : 1
- Tree Guards required                               : 1
- Maintenance Of Existing Swimming Pools             : 1
- Maintenance Of Lights In Parks                     : 1
- KSRTC - Need new Bus Route                         : 1
- Construction Of Educational Institutions           : 1
- Maintenance Of Existing Water Tank                 : 1
- Maintenance Of Lights In Playground                : 1
- Construction of skywalks                           : 1
- Increase public taps and borewells                 : 1
- Maintenance Of Existing Playground                 : 1
- 1C- Repair Dirty or Unusable Public Toilet         : 1

Key Subcategory Characteristics:
- Over 85 subcategories have fewer than 5 records.
- Semantic fragmentation: Multiple names for identical issues (e.g. 'Fixing/Reparing Potholes' vs 'Potholes').
- Prefix codes present in raw data: e.g. '1A- Address Public Urination', '2B- Repair Broken Footpath'.
- Case mismatches: 'Wrong Parking' (112) vs 'Wrong parking' (1).

================================================================================
7. LOCATION ANALYSIS
================================================================================
- Field 'location'   : 0 missing (100% present). Contains geocoded address strings (12,096 unique).
- Field 'address'    : 7,193 missing (44.76%). When provided, contains landmark text (8,069 unique).
- Field 'ward_id'    : 0 missing (100% present). 198 unique municipal wards.
- Field 'ward_title' : 32 missing (0.20%). 199 unique ward names.
- Field 'latitude'   : 0 missing (100% present). Valid WGS84 floats.
- Field 'longitude'  : 0 missing (100% present). Valid WGS84 floats.

Geospatial Bounding Box Verification:
- Latitude Range  : [12.7122605, 13.1827557]
- Longitude Range : [77.4309412, 77.80937426]
- Corrupted / Zero Coordinates: 0 (0.00%)
- Out-of-bounds Coordinates  : 0 (0.00%)
All 16,071 records fall strictly inside the Bengaluru municipal boundaries. Ready for PostGIS Point geometry.

================================================================================
8. DATE / TIME ANALYSIS
================================================================================
- Date Field: 'created_at'
- Missing / Null Dates: 0 (100% populated)
- Minimum Timestamp: 2019-01-01 06:33:00 (Jan 1, 2019)
- Maximum Timestamp: 2022-12-07 16:37:00 (Dec 7, 2022)
- Total Time Span: ~4 continuous years of municipal data.
- Date Format Variations:
  * Hyphen format: D-M-YYYY HH:MM (e.g., '1-1-2019 06:33')
  * Slash format : M/D/YYYY HH:MM (e.g., '3/31/2019 18:18')
- Parse Success: 16,071 / 16,071 (100% valid parsed dates).

================================================================================
9. STATUS AND DEPARTMENT / AGENCY ANALYSIS
================================================================================
Complaint Status ('complaint_status_title'):
- 'Open'       : 8,817 (54.86%)
- 'Resolved'   : 4,937 (30.72%)
- 'On-the-Job' : 1,653 (10.29%)
- 'Re-opened'  :   332  (2.07%)
- 'Rejected'   :   220  (1.37%)
- 'Closed'     :   112  (0.70%)

Civic Agency ('civic_agency_title'):
- 'BBMP'                                     : 12,824 (79.80%)
- 'BTP' (Traffic Police)                     :    946  (5.89%)
- 'BWSSB' (Water & Sewerage)                 :    657  (4.09%)
- 'NULL' (Missing)                           :    655  (4.08%)
- 'KSPCB' (Pollution Control)                :    432  (2.69%)
- 'BESCOM' (Electricity)                     :    240  (1.49%)
- 'BCP' (City Police)                        :    146  (0.91%)
- 'Bruhat Bengaluru Mahanagara Palike' (BBMP):    101  (0.63%)
- 'BMTC' (Buses)                             :     37  (0.23%)
- 'KSFES' (Fire Service)                     :     10  (0.06%)
- 'Bangalore Traffic Police' (BTP)           :      7  (0.04%)
- 'Karnataka State Pollution Control Board'  :      5  (0.03%)
- 'KSRTC'                                    :      4  (0.02%)
- 'BDA' (Development Authority)              :      4  (0.02%)
- 'Bangalore Water Supply And Sewerage Board':      2  (0.01%)
- 'Bangalore Electricity Supply Company'     :      1  (0.01%)

Citizen Engagement ('comment_count'):
- Min: 0, Max: 115, Mean: 1.26
- Complaints with >= 1 comments: 8,293 (51.60%)

================================================================================
10. DATA QUALITY SUMMARY
================================================================================
1. Missing Data:
   - 'address': 44.76% missing (landmark text).
   - 'civic_agency_title': 4.08% missing.
   - 'category_title', 'ward_title', 'category_id': 0.20% (32 records).
   - 'sub_category_title', 'sub_category_id': 0.11% (18 records).
   - Core fields ('title', 'description', 'location', 'latitude', 'longitude'): 0% missing.

2. Duplicate Data:
   - 33 identical duplicate rows.
   - 586 identical (title, description) text pairs.

3. Suspicious / Non-Civic Values:
   - Several categories and subcategories do not belong to municipal infrastructure:
     e.g., Police traffic violations (Helmet, Parking, Eve Teasing), Animal birth control, Ration Cards, Pensions.
   - Reinforces the need for Phase 2.5 (Civic vs. Non-Civic classification).

4. Fields Useful for Downstream Tasks:
   - For NLP Classification: 'title', 'description', combined full text.
   - For Duplicate Detection: 'latitude', 'longitude', 'created_at', 'title', 'description'.
   - For Priority Scoring: 'comment_count', emergency text keywords, duplicate cluster volume, resolution status.

================================================================================
11. FINAL PHASE 2.1 SUMMARY & ROADMAP FOR PHASE 2.2
================================================================================
What the dataset contains:
16,071 real-world urban municipal complaints across 198 wards of Bengaluru over a 4-year period (2019-2022) with 100% complete geocoded coordinates.

Important problems found:
1. Taxonomy Fragmentation: 220 raw subcategories and 44 raw categories with massive long-tail sparsity.
2. Presence of Non-Civic records that need separate labeling.
3. 33 exact duplicate rows and mixed datetime formatting conventions.

What needs to be addressed in Phase 2.2 (Data Cleaning):
1. Remove the 33 exact duplicate rows.
2. Standardize datetime strings to ISO format (YYYY-MM-DD HH:MM:SS).
3. Construct the combined text field (full_complaint = title + "\n" + description).
4. Impute or cleanly handle the 32 records with missing category/ward labels.
5. Prepare normalized records for Phase 2.3 (mapping 220 raw subcategories -> 24 target subcategories).
