import json
import os
import pandas as pd



def extract_json_document(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def extract_authors(data):
    list_of_authors = []
    authors = data['authors']
    for author in authors:
        author = f"{author['first']} {author['last']}"
        list_of_authors.append(author)
    return list_of_authors


def extract_author_info(data):
    authors = data['authors']
    extracted_data = []

    for author in authors:
        info = {
            'first_name': author.get('first', ''),
            'last_name': author.get('last', ''),
            'laboratory': author.get('affiliation', {}).get('laboratory', ''),
            'institution': author.get('affiliation', {}).get('institution', ''),
            'location_settlement': author.get('affiliation', {}).get('location', {}).get('settlement', ''),
            'location_region': author.get('affiliation', {}).get('location', {}).get('region', ''),
            'email': author.get('email', '')
        }
        extracted_data.append(info)

    return extracted_data


def extract_document_metadata(data):
    pdf_parse = data['pdf_parse']
    doi = pdf_parse['doi']
    body_text = pdf_parse['body_text']
    document_section = dict()
    section = body_text[0]['section']
    document = ''
    for text in body_text:
        if text['section'] == section:
            document += text['text']
            if section == '':
                temp_section = "plain_text"
                document_section[temp_section] = document
            else:
                document_section[section] = document
        else:
            document = ''
            document += text['text']
            section = text['section']
            document_section[section] = document

    abstract = pdf_parse['abstract']
    keywords = pdf_parse['keywords']
    bib_entries = pdf_parse['bib_entries']
    return doi, document_section, abstract, keywords


folder_path = "data/assignementdataset/"

df = pd.DataFrame({
    "Document Name": [],
    "Author First Name": [],
    "Author Last Name": [],
    "Laboratory": [],
    "Institution": [],
    "Location Settlement": [],
    "Region Settlement": [],
    "Email": [],
    "DOI": [],
    "Title": [],
    "Section": [],
    "Document Content": [],
    "Abstract": [],
    "Keywords": []
})

document_names = os.listdir(folder_path)

for doc in document_names:
    file_path = os.path.join(folder_path, doc)
    print(f"Processing file: {file_path}")

    try:
        data = extract_json_document(file_path)
        title = data['title']
        list_of_authors = extract_author_info(data) or []
        print(f"Authors are: {len(list_of_authors)}")

        doi, document_section, abstract, keywords = extract_document_metadata(data)
        document_section = document_section or {}
        abstract_text = abstract[0]['text'] if abstract and len(abstract) > 0 else ""
        keywords_str = ", ".join(keywords) if keywords else ""

        # Iterate through authors and sections
        for author in list_of_authors:
            author_info = {
                "first_name": author.get("first_name", ""),
                "last_name": author.get("last_name", ""),
                "laboratory": author.get("laboratory", ""),
                "institution": author.get("institution", ""),
                "location_settlement": author.get("location_settlement", ""),
                "location_region": author.get("location_region", ""),
                "email": author.get("email", "")
            }

            for section, content in document_section.items():
                print(f"Iterating row {df.shape[0]}")
                df.loc[df.shape[0]] = [
                    str(doc), author_info["first_name"], author_info["last_name"], author_info["laboratory"],
                    author_info["institution"],
                    author_info["location_settlement"], author_info["location_region"], author_info["email"], doi,
                    title,
                    section, content, abstract_text, keywords_str
                ]
                print(f"Shape of the DataFrame for section '{section}' and author '{author_info}': {df.shape}")

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")


def aggregate_authors(group):
    authors = []
    for _, row in group.iterrows():
        authors.append({
            "First Name": row["Author First Name"],
            "Last Name": row["Author Last Name"],
            "Institution": row["Institution"],
            "Location Settlement": row["Location Settlement"],
            "Region Settlement": row["Region Settlement"],
            "Email": row["Email"]
        })
    return authors


# Group by DOI to consolidate papers
grouped = df.groupby("DOI").agg({
    "Document Name": "first",
    "Document Content": lambda x: " ".join(x),
    "Abstract": "first",
    "Keywords": "first",
    "Title": "first",
    "Author First Name": list,
    "Author Last Name": list,
    "Institution": list,
    "Location Settlement": list,
    "Region Settlement": list,
    "Email": list
})

# Transform grouped authors' data into nested structures
grouped["Authors"] = df.groupby("DOI").apply(aggregate_authors)

# Drop individual author-related columns, as they're now in `Authors`
final_df = grouped.drop(columns=["Author First Name", "Author Last Name",
                                 "Institution", "Location Settlement",
                                 "Region Settlement", "Email"]).reset_index()

data = final_df.copy()

data.to_csv("final.csv", index=False)