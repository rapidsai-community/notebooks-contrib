import numpy as np
import matplotlib.pyplot as plt

def scatter(x,y,values,xlabel='x',ylabel='y',title=None):
    """
    Builds a scatter plot specific to the LSST data format by plotting 
    the flux over time for different passbands for a single object.
    """
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    colors = np.array([colors[i] for i in values])
    ps = []
    bs = []
    bands = ['passband_%s'%i for i in ['u', 'g', 'r', 'i', 'z','y']]
    for i in sorted(np.unique(values)):
        mask = values==i
        if len(x[mask]):
            p = plt.scatter(x[mask],y[mask],c=colors[mask])
            ps.append(p)
            bs.append(bands[i])
    plt.legend(ps,bs,scatterpoints=1)
    if title is not None:
        plt.title(title)
    
    plt.xlim([np.min(x)-10,np.min(x)+1500])
    plt.ylabel('y: %s'%ylabel)
    plt.xlabel('x: %s'%xlabel)
    
    
def groupby_aggs(df,aggs,col = "object_id"):
    """
    Given a Dataframe and a dict of {"field":"["agg", "agg"]"}, perform the 
    given aggregations using the given column as the groupby key. The original
    (non-aggregated) field is dropped from the dataframe. 
    """

    res = None
    for i,j in aggs.items():
        for k in j:
            tmp = df.groupby(col,as_index=False).agg({i:[k]})
            tmp.columns = [col,'%s_%s'%(k,i)]
            res = tmp if res is None else  res.merge(tmp,on=[col],how='left')
        df.drop_column(i)
    return res
