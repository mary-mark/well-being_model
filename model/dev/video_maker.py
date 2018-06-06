import moviepy.editor as mpy 
import numpy as np

#movie_array = []
#for nfood, dfood in enumerate(np.linspace(0.00,1.00,21)):
#    dfood_str = str(5*int(nfood))

#    movie_array.append('../output_plots/FJ/sectoral/food_inc'+dfood_str+'_vs_income.png')

#my_clip = mpy.ImageSequenceClip(movie_array, fps=2.0)
#my_clip.write_gif('../output_plots/FJ/sectoral/food_inc_vs_income.gif')

file_dir = '/Users/brian/Desktop/BANK/hh_resilience_model/output_plots/PH/png/'
for haz in ['PF','HU','EQ']:
    file_list = []
    for rp in ['1','10','25','30','50','100','200','250','500','1000']:
        file_list.append(file_dir+'npr_poverty_k_NCR_'+haz+'_'+rp+'.png')

    my_clip = mpy.ImageSequenceClip(file_list, fps=1.5)
    my_clip.write_gif(file_dir+'_gif_poverty_'+haz+'.gif')
    
    #myclip = mpy.ImageClip(file_list).write_gif(file_dir+'poverty_eq.gif')
