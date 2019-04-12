# LANL WEL (Windows Event Log) Raw Data Generator

The purpose of this script is to generate raw logs from the parsed [LANL 2017](https://csr.lanl.gov/data/2017.html) json data. The intent is to use the raw data to demonstrate parsing capabilities using cuDF. To generate the raw logs, we have templated windows event raw log data for a group of event codes.

## WEL Log Templates

The templated logs were gathered from [Ultimate Windows Security](https://www.ultimatewindowssecurity.com/securitylog/encyclopedia/)

For example, the `raw_data_generator/templates/event_4624.txt` file was templated from the example log provided [here](https://www.ultimatewindowssecurity.com/securitylog/encyclopedia/event.aspx?eventid=4624#examples).

To add a template for a new windows log event type add it to the `raw_data_generator/templates` directory. The templates use [Jinja2](http://jinja.pocoo.org/docs/2.10/) annotation and will be populated using the json property names of the parsed LANL 2017 json data.

## Requirements
1. Python (tested on 3.6)

## Run Data Generator

```$xslt
$ python run_raw_data_generator.py --help
usage: run_raw_data_generator.py [-h] input_file output_directory

LANL Raw Data Generator

positional arguments:
  input_file        Full path to input data source file
  output_directory  Full path to desired output directory

optional arguments:
  -h, --help        show this help message and exit

```

## Run Unit Test

```$xslt
cd <path-to>/raw_data_generator
python -m unittest
```
