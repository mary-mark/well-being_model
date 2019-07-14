from libraries.lib_country_dir import *

def get_asset_infos(myC,cat_info,hazard_ratios,df_haz):
    economy     = get_economic_unit(myC)
    event_level = [economy, 'hazard', 'rp']

    if myC == 'FJ':
        
        hazard_ratios_infra = get_infra_destroyed(myC,df_haz)
        
        hazard_ratios_infra = pd.merge(hazard_ratios_infra.reset_index(),hazard_ratios['fa'].reset_index(),on=[economy,'hazard','rp'],how='outer')
        hazard_ratios_infra = hazard_ratios_infra.set_index(['sector']+event_level+['hhid'])
        hazard_ratios_infra['v_k'] = hazard_ratios_infra['frac_destroyed']/hazard_ratios_infra['fa']
        
        ##adds the hh_share column in cat_info. this is the share of household's capital that is not infrastructure
        cat_info['hh_share'] = 1-hazard_ratios_infra.share.sum(level=[economy,'hazard','rp','hhid']).mean()
    
        # Adds the public_loss variable in hazard_ratios
        # This is the share of households's capital that is destroyed and does not directly belongs to the household 
        # --> fa is missing but it's the same for all capital
        hazard_ratios['public_loss_v'] = hazard_ratios_infra[["share","v_k"]].prod(axis=1, skipna=True).sum(level=event_level+['hhid'])

        #Calculation of d(income) over dk for the macro_multiplier. will drop all the intermediate variables at the end
        if False:
        #no fancy macro_multiplier for now. need to update for all sectors.
            service_loss        = get_service_loss(myC)
            service_loss_event  = pd.DataFrame(index=service_loss.unstack('sector').index) #removes the sector level
            service_loss_event['v_product'] = ((1-service_loss.cost_increase)**service_loss.e).sum(level=['hazard','rp'])
            service_loss_event['alpha_v_sum'] = hazard_ratios[['fa','v','k','pcwgt']].prod(axis=1).sum(level=['hazard','rp'])/hazard_ratios[['k','pcwgt']].prod(axis=1).sum(level=['hazard','rp']) 
            # ^ fraction of assets lost at national level
            service_loss_event['avg_prod_k'] = df.avg_prod_k.mean()
            service_loss_event["dy_over_dk"]  = ((1-service_loss_event['v_product'])/service_loss_event['alpha_v_sum']*service_loss_event["avg_prod_k"]+service_loss_event['v_product']*service_loss_event["avg_prod_k"]/3)
            service_loss_event["dy_over_dk"] = service_loss_event[["dy_over_dk",'avg_prod_k']].max(axis=1)

            hazard_ratios = pd.merge(hazard_ratios.reset_index(),service_loss_event.dy_over_dk.reset_index(),on=['hazard','rp'],how='inner')
            hazard_ratios['dy_over_dk'] = hazard_ratios['dy_over_dk'].fillna(df.avg_prod_k.mean())
    
        hazard_ratios["dy_over_dk"] = get_avg_prod(myC)
        
        hazard_ratios = hazard_ratios.drop(['k','pcwgt'],axis=1)

    elif myC == 'PH':
        # hh_share is already in hazard_ratios
        # public_loss_v is the same as v
        hazard_ratios['public_loss_v'] = hazard_ratios['v']
        hazard_ratios["dy_over_dk"] = get_avg_prod(myC)
        
    else:
        cat_info['hh_share'] = 1
        hazard_ratios['public_loss_v'] = 0
        hazard_ratios["dy_over_dk"] = get_avg_prod(myC)
    
    return cat_info,hazard_ratios
