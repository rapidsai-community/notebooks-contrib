# Introduction to cuDF using the California Housing Dataset

## Goals

In this notebook we outline an introduction into using basic functions of the cuDF library. We select a specific area in a housing dataset to generate the average number of bedrooms/rooms per household in that area.  This is the companion notebook for my blog on [Getting Started with cuDF](https://medium.com/@darrenramsook/getting-started-with-cudf-rapids-48a4b5b09b00)

The main _learning_ outcomes are defined as follows:

- Read data from a .csv file directly or from an existing pandas dataframe.
- Selection over certain rows/columns on a cuDF dataframe.
- Running queries to filter data points on a cuDF dataframe.
- Performing dataframe-wise mathematical operations on a cuDF dataframe.
- Creating a script based on cuDF functions for automation purposes.
- Provide an understanding of using basic cuDF functions in a real-world application.

The _workflow_ of the script, which satisfies the above goals, is as follows:

1. Importing the Califronia housing data. (`cudf.read_csv` OR `cudf.DataFrame.from_pandas`)
2. Selecting relevant columns across all rows of the generated dataframe.
   (`cudf.dataframe.DataFrame.loc`)
3. Performing filter based queries to isolate a geographic area. (`cudf.dataframe.DataFrame.query`)
4. Evaluating the average number of rooms in the defined area. (`cudf.dataframe.DataFrame.mean` AND `cudf.dataframe.series.Series.sum`)

## Data

The data being used is the [California Housing Prices dataset](https://www.kaggle.com/camnugent/california-housing-prices). This dataset consists of the housing features across California districts from the 1990 Census and is available to the public under the CC0 license.

## Acknowledgements

Notebook created by [Darren Ramsook](http://www.darrenr.co.tt/).
