#!/usr/bin/python3

# Colab/Jupyter/IPython notebooks formatter
# Can fix indentation and clear cells' output, useful for readability and versioning

# MIT License - Copyright (C) 2019 Filippo Rigotto
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from collections import OrderedDict
import argparse, json, os

def main():
    parser = argparse.ArgumentParser(description='Colab/Jupyter/IPython notebooks formatter')
    parser.add_argument('notebook', help='notebook (ipynb) file to process')
    parser.add_argument('-i', '--indent', type=int, default=2, help='indentation width (default: 2, negative: single line)')
    parser.add_argument('-o', '--output', type=str, default='', help='output file name (default: empty, file is overwritten)')
    parser.add_argument('-c', '--clean_output', default=False, action='store_true', help='whether to remove cells\' output')
    parser.add_argument('-s', '--dry_run', default=False, action='store_true', help='simulation: only prints arguments')
    args = parser.parse_args()

    if args.dry_run:
        print(args)
        return

    inplace = False
    if args.output == '':
        # cannot directly overwrite a file
        args.output = args.notebook + '-temp'
        inplace = True

    if args.indent < 0:
        args.indent = None

    with open(args.notebook, 'r', encoding='utf8') as in_file:
        with open(args.output, 'w', encoding='utf8') as out_file:
            notebook = json.load(in_file, object_pairs_hook=OrderedDict)

            # outputs cleaning
            if args.clean_output:
                for cell in notebook['cells']:
                    if cell['cell_type'] == 'code':
                        cell['execution_count'] = 0
                        cell['outputs'] = []
                    if 'colab' in cell['metadata']:
                        cell['metadata']['colab'] = {}
                    if 'executionInfo' in cell['metadata']:
                        del cell['metadata']['executionInfo']
                    if 'outputId' in cell['metadata']:
                        del cell['metadata']['outputId']

            json.dump(notebook, out_file, indent=args.indent)
            out_file.write("\n") # not added by dump method

    if inplace:
        os.replace(args.output, args.notebook)

if __name__ == '__main__':
    main()
