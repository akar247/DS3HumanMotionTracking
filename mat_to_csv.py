from scipy import io as spio
import pandas as pd


annots = spio.loadmat('images.mat')
DF = pd.DataFrame(index=range(0,len(annots['RELEASE'][0][0][0][0])),columns=['Image','Category','Activity'])
DF.dataframeName = 'images.mat'

for i in range(0,len(annots['RELEASE'][0][0][0][0])):
    DF.loc[[i,0],'Image'] = annots["RELEASE"]["annolist"][0,0][0][i]['image']['name'][0, 0][0] 
    try:
        DF.loc[[i,1],'Category'] = annots["RELEASE"]["act"][0,0][:,0][i]["cat_name"][0]

    except Exception: pass

    try: 
        
        DF.loc[[i,2],'Activity'] = annots["RELEASE"]["act"][0,0][:,0][i]["act_name"][0]
    except Exception: pass
        
    DF.loc[[i,3],'Train'] = annots["RELEASE"]["img_train"][0,0][0][i]

DF.to_csv('images.csv')