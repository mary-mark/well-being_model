import csv
import pandas as pd

reg_dict = pd.read_csv('../inputs/PH/prov_to_reg_dict.csv',usecols=['province','region'],index_col='region')
reg_dict['province']=reg_dict['province'].str.upper()
#reg_dict['province']=reg_dict['province'].str.replace(' ','_')

reg_dict = reg_dict.reset_index().set_index('province')

#for icol in reg_dict.columns:
#    reg_dict[icol]=reg_dict[icol].str.lower()

reg_dict = reg_dict.reset_index().set_index('province').to_dict(orient='dict')


fout = open('../map_files/PH/BlankSimpleMapRegional.svg','w')
with open('../map_files/PH/BlankSimpleMap.svg') as f:

    reader = csv.reader(f,delimiter='%')# This character is not in file

    for row in reader:
        _row = row[0]
        
        _class = _row[_row.find('class='):_row.find('d=')][:-2]

        _prov  = _class[_class.find('='):][2:]

        if _prov in reg_dict['region']:
            _row = _row.replace(_prov,str(reg_dict['region'][_prov]))
        elif 'ncr' in _prov.lower() or 'manila' in _prov.lower():
            _row = _row.replace(_prov,'Ncr')
        elif 'cotabato city' in _prov.lower():
            _row = _row.replace(_prov,'XII - SOCCSKSARGEN')
        elif 'davao oriental' in _prov.lower():
            _row = _row.replace(_prov,'XI - Davao')
        elif 'isabela city' in _prov.lower():
            _row = _row.replace(_prov,'IX - Zamboanga Peninsula')

        fout.write(_row+'\n')

fout.close()
