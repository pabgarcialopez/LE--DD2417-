#! /bin/sh
python random_indexing.py -c -co ./cleaned_example.txt
diff --strip-trailing-cr ./correct_cleaned_example.txt ./cleaned_example.txt