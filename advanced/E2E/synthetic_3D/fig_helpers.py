import pandas as pd
import numpy as np
import matplotlib.pylab as plt

'''
source data 1: kaggle survey 2017 
source data 2: kaggle survey 2018
source data 2: kaggle kernels -- https://raw.githubusercontent.com/adgirish/kaggleScape
'''

def set_rcParams():
    plt.rcParams.update({'font.size': 15})
    plt.rcParams.update({'figure.figsize': (15,9)})
    plt.rcParams.update({'figure.subplot.top': .99})
    plt.rcParams.update({'figure.subplot.left': .01})
    plt.rcParams.update({'figure.subplot.right': .99})
  
def pie_plot_important_activities_2018(): # pie_plot_important_activities_2018(kaggleSurveyDF_2018, schemaSurveyDF_2018 ):
    responseCountsList = []    
    responseOptionsTextListAnnotated = ['Analyze and understand data',
                                        'Use ML services to improve product or wokflows',
                                        'Build/run data infrastructure',
                                        'Apply ML to new areas and build prototypes',
                                        'Research/advance state of the art in ML',
                                        'Other / None of these activities are important to my role']

    responseCountsList = [9532, 5481, 5233, 7233, 4934, 4663]
    
    plt.figure()
    plt.pie( x = responseCountsList,
            labels = responseOptionsTextListAnnotated,
            startangle=90,
            autopct='%1.1f%%',
            wedgeprops = { 'linewidth' : 1 , 'edgecolor' : 'white'},
            colors=np.array(plt.get_cmap('tab20').colors)
        );
    plt.title('What activities make up an important part of your role at work?')
    plt.show();    
    
    
def bar_plot_question_2017_datasize ():
    
    uniqueCounts = pd.Series( data = [ 196,  192,  688, 1355, 1737, 1428,  803,  405,  170,   67], 
                         index = ['<1MB', '1MB', '10MB', '100MB','1GB', '10GB',  '100GB', '1TB', '10TB', '100TB'])
    titleStr = "Of the models you've trained at work, what is the typical size of datasets used?"
    
    uniqueCounts.plot.bar(alpha=1, color='tab:purple', rot=90);
    plt.gca().set_axisbelow(True)
    plt.grid(True, linestyle='--', color='silver', linewidth=2)# plt.grid(True)    
    plt.gca().xaxis.grid(False)
    
    plt.title(titleStr)
    
def bar_plot_question_2017_methods ():
    uniqueCounts = pd.Series( 
        data = [5022, 4291, 3868, 3695, 3454, 3153, 2811, 2789, 2624, 2405, 2056, 
                2050, 1973, 1949, 1936, 1913, 1902, 1557, 1417, 1398, 1158, 1146, 891,  851,  793], 
        index = ['Data Visualization', 'Logistic Regression', 'Cross-Validation',
                'Decision Trees', 'Random Forests', 'Time Series Analysis',
                'Neural Networks', 'PCA and Dimensionality Reduction',
                'kNN and Other Clustering', 'Text Analytics', 'Ensemble Methods',
                'Segmentation', 'SVMs', 'Natural Language Processing', 'A/B Testing',
                'Bayesian Techniques', 'Naive Bayes', 'Gradient Boosted Machines',
                'CNNs', 'Simulation', 'Recommender Systems', 'Association Rules',
                'RNNs', 'Prescriptive Modeling', 'Collaborative Filtering'])
    titleStr = "Of the models you've trained at work, what is the typical size of datasets used?"
    
    uniqueCounts.plot.bar(alpha=1, color='tab:purple', rot=90);
    plt.gca().set_axisbelow(True)
    plt.grid(True, linestyle='--', color='silver', linewidth=2)# plt.grid(True)    
    plt.gca().xaxis.grid(False)
    titleStr= "What methods do you use most often?"
    plt.title(titleStr)
    
def bar_plot_2018_language():
    plt.figure()
    
    valueCounts = [8180, 2046, 1211,  903,  739,  432,  408,  355,  228,  191,  135, 117,  106,   59,   55,   46,   11]
    textOptions = ['Python', 'R',  'SQL', 'Java', 'C/C++', 'C#/.NET', 'Javascript/Typescript', 'MATLAB', 'SAS/STATA', 
                   'PHP', 'Visual Basic/VBA', 'Other', 'Scala', 'Bash', 'Ruby', 'Go', 'Julia']

    uniqueCounts = pd.Series( data= valueCounts, index = textOptions).plot.bar(alpha=1, color='tab:purple', rot=90);  

    plt.gca().set_axisbelow(True);
    plt.grid(True, linestyle='--', color='silver', linewidth=2);# plt.grid(True)    
    plt.gca().xaxis.grid(False);
    titleStr= 'What specific programming language do you use most often?';
    plt.title(titleStr);
    
    plt.show()

def display_selected_figure(figure_choice):
    set_rcParams()
    if figure_choice == 'activity breakdown':
        pie_plot_important_activities_2018();
        ''' pie_plot_important_activities_2018(kaggleSurveyDF_2018, schemaSurveyDF_2018 ); '''
        
    elif figure_choice == 'datasize':
        bar_plot_question_2017_datasize ();
        '''
        bar_plot_question(kaggleSurveyDF, schemaSurveyDF, 119, nTopResults = 10, 
                            newIndex = ['<1MB', '1MB', '10MB', '100MB','1GB', '10GB',  '100GB', '1TB', '10TB', '100TB']);
        '''
        
    elif figure_choice == 'methods used':
        bar_plot_question_2017_methods();
        '''bar_plot_question(kaggleSurveyDF, schemaSurveyDF, 180, 25);'''
        
    elif figure_choice == 'language used':
        bar_plot_2018_language()
        '''bar_plot_2018(kaggleSurveyDF_2018, schemaSurveyDF_2018, 17);'''
        
    else:
        pass
    plt.show()

    
#----------------------------------------------------------------------------------------------    
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
''' 
the code below was the original version used which actually read in kaggle survey data
it has been replaced with the hard coded version to minimize external dependencies and downloads
'''
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

'''
schemaFilePath_2018 = 'kaggle_data/2018/SurveySchema.csv'
schemaSurveyDF_2018 = pd.read_csv(schemaFilePath_2018)

surveyFilePath_2018 = 'kaggle_data/2018/multipleChoiceResponses.csv'
kaggleSurveyDF_2018 = pd.read_csv(surveyFilePath_2018, encoding="ISO-8859-1", low_memory=False)

schemaFilePath = 'kaggle_data/2017/schema.csv'
schemaSurveyDF = pd.read_csv(schemaFilePath)

surveyFilePath = 'kaggle_data/2017/multipleChoiceResponses.csv'
kaggleSurveyDF = pd.read_csv(surveyFilePath, encoding="ISO-8859-1", low_memory=False)

def count_responces(dataframe, targetCol):
    optionList = []
    for iRowResponceList in dataframe[targetCol].dropna().values:
        optionList += iRowResponceList.split(',')
    
    return pd.Series(optionList).value_counts()

def bar_plot_question(dataframe, schemaSurveyDF, qNum, nTopResults = None, newIndex = None):
    targetCol = schemaSurveyDF['Column'][qNum]
    uniqueCounts = count_responces( dataframe, targetCol)
    
    if nTopResults is not None:
        uniqueCounts = uniqueCounts.head(nTopResults)
    
    if newIndex is not None:        
        uniqueCounts = uniqueCounts.reindex( newIndex )
    plt.figure()
    
    uniqueCounts.plot.bar(alpha=1, color='tab:purple', rot=90);
    plt.gca().set_axisbelow(True)
    plt.grid(True, linestyle='--', color='silver', linewidth=2)# plt.grid(True)    
    plt.gca().xaxis.grid(False)
    
    plt.title(schemaSurveyDF['Question'][qNum].split('(')[0])


def bar_plot_2018(kaggleSurveyDF_2018, schemaSurveyDF_2018, qNum = 17):
    valueCounts = kaggleSurveyDF_2018['Q'+str(qNum)][1:].dropna().value_counts()
    textOptions = list(kaggleSurveyDF_2018['Q'+str(qNum)].dropna().value_counts().keys())[:-1]
    plt.figure()
    
    plt.show()
    pd.DataFrame({'label':textOptions, 'values':valueCounts}).plot.bar(alpha=1, color='tab:purple', rot=90);  
    plt.gca().set_axisbelow(True)
    plt.grid(True, linestyle='--', color='silver', linewidth=2)# plt.grid(True)    
    plt.gca().xaxis.grid(False)
    plt.title(kaggleSurveyDF_2018['Q'+str(qNum)][0].split('-')[0].strip())

'''