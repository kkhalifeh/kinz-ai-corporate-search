from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import json
import re

print("=== DEBUGGING ENV LOADING ===")
print(f"Current working directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir('.')}")

# Try to load .env file explicitly
env_loaded = load_dotenv()
print(f"load_dotenv() returned: {env_loaded}")

# Check if the environment variable is set
openai_key = os.getenv('OPENAI_API_KEY')
print(
    f"OPENAI_API_KEY from environment: {openai_key[:10] if openai_key else 'None'}...")

# Check all environment variables that start with OPENAI
openai_vars = {k: v for k, v in os.environ.items() if k.startswith('OPENAI')}
print(f"All OPENAI environment variables: {list(openai_vars.keys())}")
print("=== END DEBUGGING ===")

# Load environment variables from .env file
# load_dotenv()

# Initialize OpenAI embeddings and LLM
# Initialize OpenAI embeddings and LLM with hardcoded API key
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

# Define index name (single index for classification, address, business fields, partner nationality, and person fields)
index_name = "kinz-classification"

# Define the employee ranges for range-based logic
EMPLOYEE_RANGES = [
    "1-4", "5-10", "11-25", "26-50", "51-100", "101-250", "251-500", "501-1000", "More Than 1000"
]

# Prompt to extract key entities, address entities, business field entities, registration date, partner nationalities, person-related entities, and determine specificity
entity_extraction_prompt = ChatPromptTemplate.from_template("""
You are a Corporate Search Agent for Kinz AI. Your task is to extract key entities, address-related entities, business field-related entities, registration date, partner nationalities, and person-related entities from the user's query. Key entities are nouns or noun phrases representing industries, services, or products (e.g., "banking sector", "mobile app development"). Address entities include locations (e.g., "Amman", "Abdali") without predefining their type (city, district, subdistrict, street). Business field entities include employee ranges and business types. Person-related entities include title groups, genders, and salutations. Specificity levels (Sector, Subsector, Group, Filter) will be determined later based on search results, but provide a default level for guidance.

### User Query:
{query}

### Instructions:
1. Extract key entities from the query (e.g., "banking sector", "mobile app development", "Mr. John").
   - Identify entities representing industries, services, or products.
2. Extract address-related entities:
   - Locations (e.g., "Amman", "Abdali", "Al Shmaisani", "King Hussein St").
   - Look for indicators like "in", "on", "at", or standalone location names.
   - Label as generic "address" (e.g., "address:Abdali") without specifying city, district, subdistrict, or street; the correct type will be determined by search results.
3. Extract business field-related entities:
   - Employee ranges (e.g., "1-4", "26-50", "More Than 1000").
     - Recognize range-based conditions like "more than X", "less than X", "at least X", "at most X", "between X and Y", where X and Y are numbers or "More Than 1000".
     - Format conditions as a single string without colons (e.g., "more than 50", "between 26 and 50").
   - Business types (e.g., "Limited Liability", "Private Shareholding").
   - Look for indicators like "with", "type", or phrases like "employees", "business type".
4. Extract registration date-related entities:
   - Dates (e.g., "registered in 2009", "registered after 2010", "registered between 2005 and 2015").
   - Look for indicators like "registered", "since", "in", "after", "before", "between".
5. Extract partner nationality-related entities:
   - Nationalities (e.g., "Jordanian", "British").
   - Multiple nationalities may be specified (e.g., "Jordanian and British").
   - Look for indicators like "partner nationality", "nationality", or standalone nationality names.
6. Extract person-related entities:
   - Title groups (e.g., "Top Management", "Management", "General Staff").
   - Genders (e.g., "Male", "Female").
   - Salutations (e.g., "Mr.", "Ms.", "Dr.").
   - Look for indicators like "title", "gender", or salutations in names (e.g., "Mr. John").
   - If person-related entities are detected, the query is a "person" query.
7. Provide a default specificity level for each key entity (Sector, Subsector, Group, Filter):
   - Default to "filter" for specific subtypes or services (e.g., "islamic banks", "mobile app development").
   - Use "group" for general industry or service terms (e.g., "banks", "IT companies").
   - Use "subsector" for moderately specific terms (e.g., "software development").
   - Use "sector" for broad categories (e.g., "finance", "technology").
   - Note that final specificity will be determined based on search results across all levels.
8. Return the entities, address entities, business field entities, registration date, partner nationalities, person-related entities, and default specificity in the following format:
   ```
   Entities: entity1, entity2
   Address Entities: address:location_name
   Business Field Entities: employee_range:range_condition, employee_range_group:range_condition, business_type:type
   Registration Date: date_condition
   Partner Nationalities: nationality1, nationality2
   Person Entities: title_group:title, gender:gender, salutation:salutation
   Default Specificity: entity1:level, entity2:level
   ```
   - For employee_range and employee_range_group, ensure `range_condition` is a single string without colons (e.g., "26-50", "more than 50", "between 26 and 50").
   - For Registration Date, use formats like "in 2009", "after 2010", "between 2005 and 2015").
   - Levels: "filter" (most specific), "group", "subsector", "sector" (least specific).
   - Include only fields present in the query (e.g., exclude business type if not specified).
   - If no entities are found for a category, return an empty string for that field.

Now, extract the entities, address entities, business field entities, registration date, partner nationalities, person entities, and default specificity. Ensure the response is enclosed in triple backticks (``` ... ```).
""")

# Prompt to map entities to classification structure, address fields, business fields, registration date, partner nationalities, person fields, and format the output
prompt_template = ChatPromptTemplate.from_template("""
You are a Corporate Search Agent for Kinz AI, a CRM platform. Your task is to map extracted entities from a user query to a classification structure, address fields, business fields, registration date, partner nationalities, and person fields, and generate a JSON output. Vector search results are provided in the format "English Name | Arabic Name" across multiple classification levels (Sector, Subsector, Group, Filter) and address levels (City, District, Subdistrict, Street). Split these into English and Arabic components for classification and address fields, using only English names for business fields and person fields. Select only the most relevant matches to keep the output concise and accurate.

### Input Data
**Extracted Entities and Fields**:
{entities_and_specificity}

**Classification Vector Search Results**:
For each classification entity, matches are provided for all levels (Sector, Subsector, Group, Filter) in the format "English Name | Arabic Name". Select the single most relevant match per entity, prioritizing the level that matches the default specificity (e.g., Sector for "banking sector") and closely aligns with the query term.
{classification_results}

**Address Vector Search Results**:
For each address entity, matches are provided for all levels (City, District, Subdistrict, Street) in the format "English Name | Arabic Name". Select the single most relevant match, prioritizing the level that most closely aligns with the query term (e.g., District for "Abdali").
{address_results}

**Business Field Vector Search Results**:
For each business field (employee_range, employee_range_group, business_type, partner_nationality, title_group, gender, salutation), matches are provided in the format "English Name | Arabic Name". Use only the English name for each match, selecting the most relevant matches based on the query context. Only include matches for fields explicitly specified.
{business_field_results}

### Available Employee Ranges:
The following are all possible employee ranges:
- 1-4 | 1-4
- 5-10 | 5-10
- 11-25 | 11-25
- 26-50 | 26-50
- 51-100 | 51-100
- 101-250 | 101-250
- 251-500 | 251-500
- 501-1000 | 501-1000
- More Than 1000 | أكثر من 1000

### Instructions
1. **Categorize and Map Entities**:
   - Parse the extracted entities into categories: classification entities, address entities, business fields (employee_range, employee_range_group, business_type), registration date, partner nationalities, and person fields (title_group, gender, salutation).
   - For each category, use the corresponding vector search results to find matches. Only process fields explicitly specified in the extracted entities. Ignore unspecified fields (e.g., partner_nationality if no nationalities are extracted).

2. **Classification Mapping**:
   - For each classification entity, review matches from all levels (Sector, Subsector, Group, Filter) provided in the search results.
   - Select the single most relevant match per entity, prioritizing:
     - The level specified in the default specificity (e.g., Sector for "banking sector", Filter for "islamic banks").
     - The match that most closely aligns with the query term (e.g., "Finance & Banking" for "banking sector").
   - If no match is found at the default specificity level, select the closest match from the most specific available level (e.g., Filter over Group).
   - For each match (format: "English Name | Arabic Name"), split into English and Arabic names using " | " as the separator.
   - Add the English name to `yii_<level>_en_ss` (e.g., `yii_sector_en_ss` for Sector matches) and the Arabic name to `yii_<level>_ar_ss` (e.g., `yii_sector_ar_ss`).
   - Place the match in the output field corresponding to its level (e.g., Sector matches in `yii_sector_en_ss`, Filter matches in `yii_filter_en_ss`).
   - Ensure `en_ss` fields contain only English names and `ar_ss` fields contain only Arabic names.
   - If no relevant match is found for an entity, leave the corresponding fields empty.

3. **Address Mapping**:
   - For each address entity (labeled as "address:location_name"), review matches from all levels (City, District, Subdistrict, Street) provided in the search results.
   - Select the single most relevant match, prioritizing the level that most closely aligns with the query term (e.g., District for "Abdali", City for "Amman").
   - For the selected match (format: "English Name | Arabic Name"), split into English and Arabic names using " | ".
   - Add the English name to the corresponding `yii_<field>_en_s` (e.g., `yii_district_en_s` for District) and the Arabic name to `yii_<field>_ar_s` (e.g., `yii_district_ar_s`).
   - If no relevant match is found, leave all address fields empty.
   - Ensure `en_s` fields contain only English names and `ar_s` fields contain only Arabic names.

4. **Business Field Mapping**:
   - For each business field explicitly specified in the extracted entities:
     - **Employee Range and Employee Range Group**:
       - If a direct range is specified (e.g., "26-50"), select that exact range from the available employee ranges.
       - If a range condition is specified (e.g., "more than 50", "less than 50", "between 26 and 100"):
         - For "more than X" or "at least X", select all ranges greater than or equal to X (e.g., "more than 50" selects "51-100", "101-250", "251-500", "501-1000", "More Than 1000").
         - For "less than X" or "at most X", select all ranges less than or equal to X (e.g., "less than 50" selects "1-4", "5-10", "11-25", "26-50").
         - For "between X and Y", select all ranges between X and Y inclusive (e.g., "between 26 and 100" selects "26-50", "51-100").
       - Use the provided search results, but prioritize pre-processed ranges if available (e.g., all ranges above 50 for "more than 50").
       - From each match (format: "English Name | Arabic Name"), use only the English name.
       - Add the English names to `yii_employee_no_s` and `yii_employee_no_group_s` as a list.
     - **Business Type**:
       - Select the most relevant match from the business field vector search results.
       - Use only the English name from the match.
       - Add to `yii_business_type_s`.
     - **Partner Nationality**:
       - Only process if partner nationalities are explicitly specified in the extracted entities.
       - Select the 1-2 most relevant matches from the business field vector search results.
       - Use only the English name from each match.
       - Add to `yii_partner_nationality_ss` as a list.
       - If no nationalities are specified, set `yii_partner_nationality_ss` to an empty list.
     - **Title Group, Gender, Salutation**:
       - Select the most relevant match for each field from the business field vector search results.
       - Use only the English name from the match.
       - Add to `yii_title_group_s`, `yii_gender_s`, and `yii_salutation_s` respectively.
   - If a field is not specified or no match is found, leave it empty.

5. **Registration Date Mapping**:
   - Interpret the registration date condition:
     - For a specific year (e.g., "in 2009"), format as "2009-01-01T00:00:00Z".
     - For a condition (e.g., "after 2010", "before 2015"), format as a range (e.g., ">2010-01-01T00:00:00Z", "<2015-01-01T00:00:00Z").
     - For a range (e.g., "between 2005 and 2015"), format as "2005-01-01T00:00:00Z to 2015-12-31T23:59:59Z").
   - Add to `yii_registration_date`. If not specified, leave empty.

6. **Document Type**:
   - If person fields (title_group, gender, salutation) are present in the extracted entities, set `yii_doc_type_s` to "person".
   - Otherwise, set `yii_doc_type_s` to "business".

7. **Output Construction**:
   - Generate a JSON object with the following structure:
     {{
       "yii_doc_type_s": "business or person",
       "yii_sector_en_ss": [],
       "yii_sector_ar_ss": [],
       "yii_subsector_en_ss": [],
       "yii_subsector_ar_ss": [],
       "yii_group_en_ss": [],
       "yii_group_ar_ss": [],
       "yii_filter_en_ss": [],
       "yii_filter_ar_ss": [],
       "yii_city_en_s": "",
       "yii_city_ar_s": "",
       "yii_district_en_s": "",
       "yii_district_ar_s": "",
       "yii_subDistrict_en_s": "",
       "yii_subDistrict_ar_s": "",
       "yii_street_en_s": "",
       "yii_street_ar_s": "",
       "yii_employee_no_s": [],
       "yii_employee_no_group_s": [],
       "yii_business_type_s": "",
       "yii_registration_date": "",
       "yii_partner_nationality_ss": [],
       "yii_title_group_s": "",
       "yii_gender_s": "",
       "yii_salutation_s": ""
     }}
   - For classification fields (`yii_sector_en_ss`, `yii_subsector_en_ss`, `yii_group_en_ss`, `yii_filter_en_ss`), include only English names.
   - For classification fields (`yii_sector_ar_ss`, `yii_subsector_ar_ss`, `yii_group_ar_ss`, `yii_filter_ar_ss`), include only Arabic names.
   - For address fields (`yii_city_en_s`, `yii_district_en_s`, `yii_subDistrict_en_s`, `yii_street_en_s`), include only English names.
   - For address fields (`yii_city_ar_s`, `yii_district_ar_s`, `yii_subDistrict_ar_s`, `yii_street_ar_s`), include only Arabic names.
   - For business fields (`yii_employee_no_s`, `yii_employee_no_group_s`, `yii_business_type_s`, `yii_partner_nationality_ss`, `yii_title_group_s`, `yii_gender_s`, `yii_salutation_s`), include only English names.
   - Include only fields with relevant matches. Ensure the output is a valid JSON string.

### Notes
- Strictly include only fields explicitly specified in the extracted entities or query context.
- For classification fields, select one match per entity at the level matching the default specificity (e.g., Sector for "banking sector") unless a more specific, closely aligned match is found.
- For address fields, select one match per address entity from the most relevant level (e.g., District for "Abdali").
- For employee_range conditions, ensure all applicable ranges are selected based on the provided logic (e.g., "more than 50" includes "51-100" to "More Than 1000").
- For classification and address fields, split "English Name | Arabic Name" matches and assign English names to `en_ss` or `en_s` fields and Arabic names to `ar_ss` or `ar_s` fields.
- For business fields, use only the English name from each match, ignoring the Arabic component.
- Keep the output concise by limiting to one match per entity unless multiple matches are equally relevant.

Return the output as a valid JSON string enclosed in triple backticks (```json ... ```).
""")


def search_classification(entity, level):
    """Search Pinecone for the most relevant classification entries at the specified level."""
    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        namespace=level
    )
    # Use k=3 to retrieve relevant matches
    k = 3
    results = vector_store.similarity_search(entity, k=k)
    matches = []
    for result in results:
        # Returns the combined_name (e.g., "Finance & Banking | تمويل المصارف")
        matches.append(result.page_content)
    return matches


def search_address(address_entity, level):
    """Search Pinecone for the most relevant address entry at the specified level."""
    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        namespace=level
    )
    # Use k=1 to get the most relevant match for address fields
    results = vector_store.similarity_search(address_entity, k=1)
    matches = []
    for result in results:
        # Returns the combined_name (e.g., "Abdali | العبدلي")
        matches.append(result.page_content)
    return matches


def search_business_field(field_entity, field):
    """Search Pinecone for the most relevant business field entries at the specified field."""
    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        namespace=field
    )
    # Use k=5 for employee ranges to ensure all relevant ranges are retrieved, k=3 for partner_nationality, k=1 for others
    k = 5 if field in [
        "employee_range", "employee_range_group"] else 3 if field == "partner_nationality" else 1
    results = vector_store.similarity_search(field_entity, k=k)
    matches = []
    for result in results:
        # Returns the combined_name (e.g., "Top Management | الإدارة العليا")
        matches.append(result.page_content)
    return matches


def search_corporate_agent(query):
    """Main function to process the user query and return classification, address, business field, registration date, partner nationality, and person data using LLM."""
    # Step 1: Extract entities, address entities, business field entities, registration date, partner nationalities, person entities, and default specificity using the LLM
    entity_response = llm.invoke(entity_extraction_prompt.format(query=query))
    response_text = entity_response.content.strip()

    # Remove markdown code blocks if present
    response_text = re.sub(r'```(?:.*?)\n(.*)\n```', r'\1',
                           response_text, flags=re.DOTALL).strip()

    # Parse entities, address entities, business field entities, registration date, partner nationalities, person entities, and default specificity
    entities_match = re.search(r'Entities:\s*(.*?)\n', response_text)
    address_entities_match = re.search(
        r'Address Entities:\s*(.*?)\n', response_text)
    business_field_entities_match = re.search(
        r'Business Field Entities:\s*(.*?)\n', response_text)
    registration_date_match = re.search(
        r'Registration Date:\s*(.*?)\n', response_text)
    partner_nationalities_match = re.search(
        r'Partner Nationalities:\s*(.*?)\n', response_text)
    person_entities_match = re.search(
        r'Person Entities:\s*(.*?)\n', response_text)
    specificity_match = re.search(
        r'Default Specificity:\s*(.*?)$', response_text)

    if not entities_match or not address_entities_match or not business_field_entities_match or not registration_date_match or not partner_nationalities_match or not person_entities_match or not specificity_match:
        print(f"Error parsing entity extraction response: {response_text}")
        return {
            "yii_doc_type_s": "business",
            "yii_sector_en_ss": [],
            "yii_sector_ar_ss": [],
            "yii_subsector_en_ss": [],
            "yii_subsector_ar_ss": [],
            "yii_group_en_ss": [],
            "yii_group_ar_ss": [],
            "yii_filter_en_ss": [],
            "yii_filter_ar_ss": [],
            "yii_city_en_s": "",
            "yii_city_ar_s": "",
            "yii_district_en_s": "",
            "yii_district_ar_s": "",
            "yii_subDistrict_en_s": "",
            "yii_subDistrict_ar_s": "",
            "yii_street_en_s": "",
            "yii_street_ar_s": "",
            "yii_employee_no_s": [],
            "yii_employee_no_group_s": [],
            "yii_business_type_s": "",
            "yii_registration_date": "",
            "yii_partner_nationality_ss": [],
            "yii_title_group_s": "",
            "yii_gender_s": "",
            "yii_salutation_s": ""
        }

    entities_str = entities_match.group(1).strip()
    address_entities_str = address_entities_match.group(1).strip()
    business_field_entities_str = business_field_entities_match.group(
        1).strip()
    registration_date_str = registration_date_match.group(1).strip()
    partner_nationalities_str = partner_nationalities_match.group(1).strip()
    person_entities_str = person_entities_match.group(1).strip()
    specificity_str = specificity_match.group(1).strip()

    # Parse entities
    if not entities_str:
        entities = []
        default_specificity = {}
    else:
        entities = [entity.strip() for entity in entities_str.split(",")]
        specificity_pairs = [pair.strip()
                             for pair in specificity_str.split(",")]
        default_specificity = {}
        for pair in specificity_pairs:
            try:
                entity, level = pair.split(":", 1)
                default_specificity[entity.strip()] = level.strip()
            except ValueError as e:
                print(f"Error parsing default specificity pair '{pair}': {e}")
                continue

    # Parse address entities
    address_entities = []
    if address_entities_str:
        address_pairs = [pair.strip()
                         for pair in address_entities_str.split(",")]
        for pair in address_pairs:
            if pair:
                try:
                    field, value = pair.split(":", 1)
                    if field.strip() == "address":
                        address_entities.append(value.strip())
                except ValueError as e:
                    print(f"Error parsing address pair '{pair}': {e}")
                    continue

    # Parse business field entities
    business_field_entities = {}
    if business_field_entities_str and not business_field_entities_str.startswith("Registration Date"):
        print(f"Raw business field entities: {business_field_entities_str}")
        business_field_pairs = [pair.strip()
                                for pair in business_field_entities_str.split(",")]
        for pair in business_field_pairs:
            if pair:
                try:
                    field, value = pair.split(":", 1)
                    business_field_entities[field.strip()] = value.strip()
                except ValueError as e:
                    print(f"Error parsing business field pair '{pair}': {e}")
                    continue

    # Parse partner nationalities
    partner_nationalities = []
    if partner_nationalities_str:
        partner_nationalities = [nationality.strip(
        ) for nationality in partner_nationalities_str.split(",") if nationality.strip()]

    # Parse person entities
    person_entities = {}
    if person_entities_str:
        person_pairs = [pair.strip()
                        for pair in person_entities_str.split(",")]
        for pair in person_pairs:
            if pair:
                try:
                    field, value = pair.split(":", 1)
                    person_entities[field.strip()] = value.strip()
                except ValueError as e:
                    print(f"Error parsing person pair '{pair}': {e}")
                    continue

    # Log extracted entities and default specificity for debugging
    print(f"Extracted Entities: {entities_str}")
    print(f"Address Entities: {address_entities_str}")
    print(f"Business Field Entities: {business_field_entities_str}")
    print(f"Registration Date: {registration_date_str}")
    print(f"Partner Nationalities: {partner_nationalities_str}")
    print(f"Person Entities: {person_entities_str}")
    print(f"Default Specificity: {specificity_str}")

    # Step 2: Search Pinecone for classification entities across all levels
    classification_results = []
    for entity in entities:
        # Search all classification levels
        levels = ["sector", "subsector", "group", "filter"]
        entity_results = {"entity": entity, "matches": {}}
        for level in levels:
            results = search_classification(entity, level)
            validated_results = []
            for result in results:
                if " | " in result:
                    # Relevance check: ensure the result contains at least one query term
                    result_lower = result.lower()
                    if any(term.lower() in result_lower for term in entity.split()):
                        validated_results.append(result)
                    else:
                        print(
                            f"Discarding irrelevant classification result for '{entity}' at level '{level}': {result}")
                else:
                    print(
                        f"Invalid Pinecone classification result format for '{entity}' at level '{level}': {result}")
            # Limit to 2 matches per level
            entity_results["matches"][level] = validated_results[:2]
        classification_results.append(entity_results)

    # Format classification vector search results for the prompt and log them
    classification_results_text = ""
    for idx, result in enumerate(classification_results, 1):
        classification_results_text += f"Entity {idx}: \"{result['entity']}\"\n"
        for level in result["matches"]:
            quoted_matches = [f'"{r}"' for r in result["matches"][level]]
            classification_results_text += f"- {level.capitalize()}: {', '.join(quoted_matches)}\n"
    print(
        f"Classification Vector Search Results:\n{classification_results_text}")

    # Step 3: Search Pinecone for address entities across all levels
    address_results = []
    for entity in address_entities:
        # Search all address levels
        levels = ["city", "district", "subdistrict", "street"]
        entity_results = {"entity": entity, "matches": {}}
        for level in levels:
            matches = search_address(entity, level)
            validated_matches = []
            for match in matches:
                if " | " in match:
                    validated_matches.append(match)
                else:
                    print(
                        f"Invalid Pinecone address result format for '{entity}' at level '{level}': {match}")
            entity_results["matches"][level] = validated_matches
        address_results.append(entity_results)

    # Format address vector search results for the prompt and log them
    address_results_text = ""
    for idx, result in enumerate(address_results, 1):
        address_results_text += f"Address Entity {idx}: \"{result['entity']}\"\n"
        for level in result["matches"]:
            quoted_matches = [f'"{m}"' for m in result["matches"][level]]
            address_results_text += f"- {level.capitalize()}: {', '.join(quoted_matches)}\n"
    print(f"Address Vector Search Results:\n{address_results_text}")

    # Step 4: Search Pinecone for business field entities and person entities (exclude partner_nationality unless specified)
    business_field_results = {}
    for field in ["employee_range", "employee_range_group", "business_type", "title_group", "gender", "salutation"]:
        if field in business_field_entities:
            entity = business_field_entities[field]
            # Pre-process employee range conditions
            if field in ["employee_range", "employee_range_group"] and entity.startswith("more than "):
                try:
                    threshold = int(entity.split("more than ")[1])
                    matches = [f"{r} | {r}" for r in EMPLOYEE_RANGES if "-" in r and int(
                        r.split("-")[0]) >= threshold or r == "More Than 1000"]
                    print(
                        f"Pre-processed employee range for '{entity}': {matches}")
                except (ValueError, IndexError):
                    matches = search_business_field(entity, field)
            else:
                matches = search_business_field(entity, field)
            validated_matches = []
            for match in matches:
                if " | " in match:
                    validated_matches.append(match)
                else:
                    print(
                        f"Invalid Pinecone business field result format for '{entity}' at field '{field}': {match}")
            business_field_results[field] = validated_matches[:2]
        elif field in person_entities:
            entity = person_entities[field]
            matches = search_business_field(entity, field)
            validated_matches = []
            for match in matches:
                if " | " in match:
                    validated_matches.append(match)
                else:
                    print(
                        f"Invalid Pinecone person field result format for '{entity}' at field '{field}': {match}")
            business_field_results[field] = validated_matches[:2]

    # Only include partner_nationality if explicitly specified
    if partner_nationalities:
        matches = []
        for nationality in partner_nationalities:
            nationality_matches = search_business_field(
                nationality, "partner_nationality")
            validated_matches = []
            for match in nationality_matches:
                if " | " in match:
                    validated_matches.append(match)
                else:
                    print(
                        f"Invalid Pinecone partner nationality result format for '{nationality}': {match}")
            matches.extend(validated_matches[:2])
        business_field_results["partner_nationality"] = matches
    else:
        business_field_results["partner_nationality"] = []

    # Format business field vector search results for the prompt and log them
    business_field_results_text = ""
    for field in ["employee_range", "employee_range_group", "business_type", "title_group", "gender", "salutation"]:
        if field in business_field_results:
            matches = business_field_results[field]
            quoted_matches = [f'"{m}"' for m in matches]
            business_field_results_text += f"{field.capitalize()}: {', '.join(quoted_matches)}\n"
    print(
        f"Business Field Vector Search Results:\n{business_field_results_text}")

    # Step 5: Use LLM to map entities to classification structure, address fields, business fields, registration date, partner nationalities, person fields, and format output
    prompt = prompt_template.format(
        entities_and_specificity=f"Entities: {entities_str}\nAddress Entities: {address_entities_str}\nBusiness Field Entities: {business_field_entities_str}\nRegistration Date: {registration_date_str}\nPartner Nationalities: {partner_nationalities_str}\nPerson Entities: {person_entities_str}\nDefault Specificity: {specificity_str}",
        classification_results=classification_results_text,
        address_results=address_results_text,
        business_field_results=business_field_results_text
    )
    response = llm.invoke(prompt)

    # Extract JSON from the response (removing markdown code blocks if present)
    response_text = response.content.strip()
    json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_str = response_text  # Fallback to raw content if no markdown

    # Log the LLM response for debugging
    print(f"LLM Response:\n{response_text}")

    # Parse the JSON string
    try:
        output = json.loads(json_str)
        return output
    except json.JSONDecodeError as e:
        # Log the error and return a default output
        print(f"Error parsing LLM response as JSON: {e}")
        print(f"Raw response: {response_text}")
        return {
            "yii_doc_type_s": "business",
            "yii_sector_en_ss": [],
            "yii_sector_ar_ss": [],
            "yii_subsector_en_ss": [],
            "yii_subsector_ar_ss": [],
            "yii_group_en_ss": [],
            "yii_group_ar_ss": [],
            "yii_filter_en_ss": [],
            "yii_filter_ar_ss": [],
            "yii_city_en_s": "",
            "yii_city_ar_s": "",
            "yii_district_en_s": "",
            "yii_district_ar_s": "",
            "yii_subDistrict_en_s": "",
            "yii_subDistrict_ar_s": "",
            "yii_street_en_s": "",
            "yii_street_ar_s": "",
            "yii_employee_no_s": [],
            "yii_employee_no_group_s": [],
            "yii_business_type_s": "",
            "yii_registration_date": "",
            "yii_partner_nationality_ss": [],
            "yii_title_group_s": "",
            "yii_gender_s": "",
            "yii_salutation_s": ""
        }
