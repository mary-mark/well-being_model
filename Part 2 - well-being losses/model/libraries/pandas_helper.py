import pandas as pd   
 
# from sorted_nicely import sorted_nicely

 
def get_list_of_index_names(df):
    """returns name of index in a data frame as a list. (single element list if the dataframe as a single index)""
    """
    
    if df.index.name is None:
        return list(df.index.names)
    else:
        return [df.index.name] #do not use list( ) as list breaks strings into list of chars

 
    
def broadcast_simple( df_in, index):    
    """simply replicates df n times and adds index (where index has n distinct elements) as the last level of a multi index.
    if index is a multiindex with (m,p) this will work too (and replicte df n=m *p times). But if some of the levels of index are already included in df_in (BASED ON NAME ONLY), these are ignored (see example).
        
    EXAMPLES
    
    s=pd.DataFrame(["a","b","c"], index=pd.Index(["A", "B", "C"], name="letters"), columns=["value"])
    s
    
        value
    A 	a
    B 	b

    #works
    my_index=pd.Index(["one", "two"], name="numbers")
    broadcast_simple(s, my_index)

       numbers
    A  one        a
       two        a
    B  one        b
       two        b
   Name: value, dtype: object
    
    #multi index example
    my_index=pd.MultiIndex.from_product([["one", "two"], ["cat", "dog"]], names=["numbers", "pets"])
    broadcast_simple(s, my_index)

       numbers  pets
    A  one      cat     a
                dog     a
       two      cat     a
                dog     a
    B  one      cat     b
                dog     b
       two      cat     b
                dog     b
    Name: value, dtype: object
    
    #Ignored level in multi index example
    my_index=pd.MultiIndex.from_product([["one", "two"], ["X", "Y"]], names=["numbers", "letters"])
    broadcast_simple(s, my_index)
    
    letters  numbers
    A        one        a
             two        a
    B        one        b
             two        b
    C        one        c
             two        c
    

    #Raise error because the index should be named
    my_index=pd.Index(["one", "two"])
    broadcast_simple(s, my_index)
    
    """

    #in case of MultiIndex, performs this function on each one of the levels of the index
    if type(index)== pd.MultiIndex:
        y = df_in.copy()
        for idxname in [i for i in index.names if i not in get_list_of_index_names(df_in)]:
                y = broadcast_simple(y, index.get_level_values(idxname))
        return y
    
    cat_list = index.unique()
    nb_cats =len(cat_list)
    if index.name is None:
        raise Exception("index should be named")
        
    
    y= pd.concat([df_in]*nb_cats, 
                    keys = cat_list, 
                    names=[index.name]+get_list_of_index_names(df_in)
                 )
                
    #puts new index at the end            
    y=y.reset_index(index.name).set_index(index.name, append=True).sort_index()
    
    return y.squeeze()
    

def concat_categories(p,np, index):
    """works like pd.concat with keys but swaps the index so that the new index is innermost instead of outermost
    http://pandas.pydata.org/pandas-docs/stable/merging.html#concatenating-objects
    """
    
    if index.name is None:
        raise Exception("index should be named")
        
    
    y= pd.concat([p, np], 
        keys = index, 
        names=[index.name]+get_list_of_index_names(p)
            )#.sort_index()
    
    #puts new index at the end            
    y=y.reset_index(index.name).set_index(index.name, append=True).sort_index()
    
    #makes sure a series is returned when possible
    return y.squeeze()

def merge_multi(self, df, on):
    return self.reset_index().join(df,on=on).set_index(self.index.names)
