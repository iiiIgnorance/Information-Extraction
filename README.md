# Information-Extraction

## Files
Name | Description
--- | ---
proj2.tar.gz | Information Extraction System
transcript-spanbert.pdf | A transcript of the runs on spanbert test cases
transcript-gpt3.pdf | A transcript of the runs on gpt3 test cases
requirements.txt | Project dependencies
README.pdf | Project description

## Run
```bash
$ python3 project2.py [-spanbert|-gpt3] <google api key> <google engine id> <openai secret key> <r> <t> <q> <k>
```
Example:  
```bash
$ python3 project2.py -gpt3 AIzaSyCn8r6T7HFDtIroyakH0xp5T8UKOd9T2iU 2d27291d6a2f66dc1 <openai secret key> 1 0.7 "mark zuckerberg harvard" 10
```
```bash
$ python3 project2.py -spanbert AIzaSyCn8r6T7HFDtIroyakH0xp5T8UKOd9T2iU 2d27291d6a2f66dc1 <openai secret key> 1 0.7 "mark zuckerberg harvard" 10
```

Credential | Key
--- | ---
Google Api Key | AIzaSyCn8r6T7HFDtIroyakH0xp5T8UKOd9T2iU
Google Engine Id | 2d27291d6a2f66dc1


## Project Design
The system takes a query, retrieves search results from Google, extracts the actual plain text from the webpage using Beautiful Soup, uses the spaCy library to split the text into sentences and extract named entities, uses SpanBERT or  OpenAI GPT-3's API to extract at least k tuples.
1. ``get_parameters():``This function gets the input parameters from the user and prints them.
2. ``google_search():``This function retrieves search results from Google using the Google Api Key and Google Engine Id.
3. ``extract_plain_text():``This function extract the actual plain text from the webpage using ``Beautiful Soup``.
4. ``use_spanbert():``This function uses ``spaCy`` library to split the text into sentences and extract named entities and uses ``SpanBERT`` to predict the corresponding relations, and extract all instances of the relation specified by input parameter r.
5. ``use_gpt3():``This function uses the OpenAI GPT-3's API for relation extraction.
6. ``extract_relations_gpt3():`` This function is a rewrite of the ``extract_relations()`` function in ``spacy_help_functions.py`` when it is used on gpt3.

## Step 3 Description
The step 3 is implemented in the ``use_spanbert()`` and ``use_gpt3()`` functions respectively.  

First, we use ``previous_url = set()`` to save the url which have been seen, and skip the already-seen url.  

Second, we call ``extract_plain_text():`` function to extract the actual plain text. We process the discrete text into one continuous paragraph without interruptions and save at most 10,000 characters.

Third, we use ``spaCy`` library in ``use_spanbert()`` to split the text into sentences and extract named entities.(``doc = nlp(plain_text)``)

Fourth, if ``-spanbertis`` specified, we call ``extract_relations()``, which is in ``spacy_help_functions.py``, to extract the relation tuple saving in ``X``. Otherwise, if ``-gpt3`` is specified , we call ``extract_relations_gpt3()`` function to extract the relation tuple saving in ``X``.

Fifth, when saving tuples with a confidence of at least ``t`` or removing duplicate tuples, we use the methods provided in ``spacy_help_functions.py``.







