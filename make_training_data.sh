# dataset 1: 10000 programs with a variety of task lengths
python3 make_dataset.py -n 10000 -mi 40 -ma 30000 -f dataset1
# dataset 2: 100 programs with very long potential tasks
python3 make_dataset.py -n 100 -mi 300 -ma 1000 -f dataset2
# dataset 3: 500 programs with mostly short tasks arriving in quick succession
python3 make_dataset.py -n 500 -mi 5 -ma 250 -f dataset3
# dataset 4: Same as dataset 1 with chi-squared distribution
python3 make_dataset.py -n 10000 -mi 40 -ma 30000 -d cs -f dataset4
# dataset 5: Same as dataset 1 with shorter task lengths
python3 make_dataset.py -n 10000 -mi 20 -ma 30000 -f dataset5