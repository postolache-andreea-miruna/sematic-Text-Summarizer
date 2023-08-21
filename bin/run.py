#!/usr/bin/env python

import sys
from base.base_summarizer import get_summary

if __name__ == '__main__':
    model_path = sys.argv[0]
    print("Going to load word2Vec Model, it may take a minute or more")

    print("model loaded, now going to quickly generate summary")

    summary_percent = 5
    input_file = sys.argv[1]
    summary_sens = get_summary(input_file, summary_percent, model_path=model_path)
    for s in summary_sens:
        print(s)
