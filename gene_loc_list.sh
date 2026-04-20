#!/bin/sh

awk -F',' 'NR==1 {print "Chromosome:Start:End"; next} {print $3 ":" $4 ":" $5}' output/rf_best_k_val_features.csv | sed '1d' | sed 's/$/:1/' > output/rf_best_k_val_features_loc.csv

awk -F',' 'NR==1 {print "Chromosome:Start:End"; next} {print $3 ":" $4 ":" $5}' output/svm_best_k_val_features.csv
| sed '1d' | sed 's/$/:1/' > output/svm_best_k_val_features_loc.csv
