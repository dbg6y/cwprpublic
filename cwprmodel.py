### Community Water Project Model


## Import Modules
import numpy as np
import pandas as pd
import scipy.signal as signal
import sqlite3
import multiprocessing
import json
import sys
from matplotlib import pyplot as plt
import time


## Define Classes
class Precipitation(object):
    '''The seasonal rainfall parameters for both the catchment and the locality 
    
    Attributes: alph_prec, lmbd_prec, prdy_prec, rcrd_prec
    '''
    def __init__(self):
        
        # Insert placeholders for attributes
        self.alph_prec = None
        self.lmbd_prec = None
        self.prdy_prec = None 
        self.rcrd_prec = None
        
    def prec_sims(self):
        '''Simulate seasonal rainfall.'''
        
        # Define season length
        rsea_prec = INVL_SIMU * (TSEA_CLIM + BUFF_SIMU)
        dsea_prec = INVL_SIMU * int(TGRW_CROP - TSEA_CLIM)
        
        # Choose seasonal parameters
        alph_prec = np.random.gamma(1 / ARCV_CLIM**2, 
                                        ARMN_CLIM * ARCV_CLIM**2, 1)[0]  # m
        lmbd_prec = np.random.gamma(1 / LRCV_CLIM**2, 
                                    LRMN_CLIM * LRCV_CLIM**2, 1)[0]  # days-1
        prdy_prec = alph_prec * lmbd_prec  # m day-1
        
        # Simulate precipitation for rainy and dry seasons
        rrcd_prec = np.random.binomial(1, lmbd_prec / INVL_SIMU, 
                                       rsea_prec).astype(np.float)
        drcd_prec = np.random.binomial(1, LDMN_CLIM / INVL_SIMU, 
                                       dsea_prec).astype(np.float)
        ramt_prec = np.random.exponential(alph_prec, 
                                          len(rrcd_prec[rrcd_prec > 0]))
        damt_prec = np.random.exponential(ADMN_CLIM, 
                                          len(drcd_prec[drcd_prec > 0]))
        np.place(rrcd_prec, rrcd_prec == 1, ramt_prec)
        np.place(drcd_prec, drcd_prec == 1, damt_prec)
        rcrd_prec = np.concatenate((rrcd_prec, drcd_prec))
        
        # Save attributes
        self.alph_prec = alph_prec
        self.lmbd_prec = lmbd_prec
        self.prdy_prec = prdy_prec
        self.rcrd_prec = np.array(rcrd_prec, dtype = float)


class Catchment(object):
    '''The soil moisture and river flow from the catchment 
    
    Attributes: ctar_catc, ctln_catc, prec_catc, smst_catc, rnof_catc, 
    leak_catc, flow_catc
    '''
    def __init__(self):
       
        # Define static attributes
        self.ctar_catc = CAMN_CATC #np.random.gamma(1 / CACV_CATC**2, 
                                         #CAMN_CATC * CACV_CATC**2, 1)[0]  # m**2
        self.ctln_catc = 1.4 * ((self.ctar_catc / 1000**2) / 2.58999)**0.6 \
                         * 1.60934 * 1000  # m
        
        # Insert placeholders for other attributes
        self.smst_catc = None
        self.rnof_catc = None
        self.leak_catc = None
        self.flow_catc = None
    
    def _auih_catc(self, tmes_seas):
        '''Internal function to calculate the fast flow distribution'''
        ctka_catc = 1 / (CTMU_CATC * self.ctar_catc**0.38) # days**-1
        apdf_catc = ctka_catc * np.exp(-ctka_catc * tmes_seas)
        return apdf_catc
    
    def _buih_catc(self, tmes_seas):
        '''Internal function to calculate the base flow distribution'''
        ctkb_catc = 1 / (CTMU_CATC * self.ctar_catc**0.38) / 10  # days**-1
        bpdf_catc = ctkb_catc * np.exp(-ctkb_catc * tmes_seas)
        return bpdf_catc
    
    def _cuih_catc(self, tmes_seas):
        '''Internal function to calculate the channel flow distribution'''
        cpdf_catc = (self.ctln_catc / (np.sqrt(4 * np.pi * CTDC_CATC) 
                                       * tmes_seas**(3 / 2))) \
               * np.exp(-((self.ctln_catc - CTWC_CATC * tmes_seas)**2)  # days**-1
                        / (4 * CTDC_CATC * tmes_seas))
        return cpdf_catc
    
    def smst_sims(self, prec):
        '''Simulates soil moisture dynamics in the catchment '''
        
        prec_catc = prec.rcrd_prec

        # Set initial values
        psms_invl = SFLD_SOIL
        smst_catc = []
        rnof_catc = []
        leak_catc = []
        
        # Loop through intervals
        for prec_invl in prec_catc: 
            
            # Add Rainfall
            smst_invl = psms_invl + prec_invl / (PORO_SOIL * ZRCT_CROP)
            
            # Remove Surface Runoff
            if smst_invl > 1.:
                rnof_invl = smst_invl - 1.
                smst_invl = 1.
            else: 
                rnof_invl = 0.
            
            # Remove Leakage
            if smst_invl > SFLD_SOIL:
                leak_invl = (KSAT_SOIL / INVL_SIMU) \
                            * (np.exp(BETA_SOIL * (smst_invl - SFLD_SOIL)) - 1.) \
                            / (np.exp(BETA_SOIL * (1. - SFLD_SOIL)) - 1.)
                smst_invl = smst_invl - leak_invl / (PORO_SOIL * ZRCT_CROP)
            else:
                leak_invl = 0.
            
            # Remove Evapotranspiration Losses
            if smst_invl > SSTR_SOIL:
                evtr_invl = ETMX_CLIM / INVL_SIMU
            elif smst_invl > SWLT_SOIL and smst_invl <= SSTR_SOIL:
                evtr_invl = (EWLT_CLIM / INVL_SIMU) + ((ETMX_CLIM - EWLT_CLIM) / INVL_SIMU) \
                            * (smst_invl - SWLT_SOIL) / (SSTR_SOIL - SWLT_SOIL)
            elif smst_invl > SHYG_SOIL and smst_invl <= SWLT_SOIL:
                evtr_invl = (EWLT_CLIM / INVL_SIMU) * (smst_invl - SHYG_SOIL) \
                            / (SWLT_SOIL - SHYG_SOIL)
            else:
                evtr_invl = 0.
            smst_invl = smst_invl - evtr_invl / (PORO_SOIL * ZRCT_CROP)
            
            # Save interval Values
            smst_catc.append(smst_invl)
            rnof_catc.append(rnof_invl)
            leak_catc.append(leak_invl)
            
            # Update iterative values
            psms_invl = smst_invl
        
        # Save attributes
        self.prec_catc = np.array(prec_catc, dtype = float)
        self.smst_catc = np.array(smst_catc, dtype = float)
        self.rnof_catc = np.array(rnof_catc, dtype = float)
        self.leak_catc = np.array(leak_catc, dtype = float)
    
    def flow_sims(self):
        '''Simulates river flow out of the catchment'''
        
        # Create internal data structures
        lgrw_seas = INVL_SIMU * (TGRW_CROP + BUFF_SIMU)
        invl_seas = 1 + np.arange(lgrw_seas)
        apfc_seas = 1 + np.arange(APFC_CATC * lgrw_seas)
        
        # Create catchment and channel hydrographs
        auih_aprx = self._auih_catc(apfc_seas / (APFC_CATC * INVL_SIMU)) \
                    / (APFC_CATC * INVL_SIMU)
        buih_aprx = self._buih_catc(apfc_seas / (APFC_CATC * INVL_SIMU)) \
                    / (APFC_CATC * INVL_SIMU)
        cuih_aprx = self._cuih_catc(apfc_seas / (APFC_CATC * INVL_SIMU)) \
                    / (APFC_CATC * INVL_SIMU)
        
        # Calculate fast and slow responses 
        rnof_aprx = np.interp(apfc_seas / APFC_CATC, invl_seas, self.rnof_catc)
        leak_aprx = np.interp(apfc_seas / APFC_CATC, invl_seas, self.leak_catc)
        arsp_aprx = self.ctar_catc * (rnof_aprx + (1 - BFPR_CATC) * leak_aprx)
        brsp_aprx = self.ctar_catc * BFPR_CATC * leak_aprx
        
        # Calculate catchment and total flow then sub in alternative minima
        aflw_aprx = signal.fftconvolve(arsp_aprx, auih_aprx)[0:len(apfc_seas)]
        bflw_aprx = signal.fftconvolve(brsp_aprx, buih_aprx)[0:len(apfc_seas)]
        cflw_aprx = aflw_aprx + bflw_aprx
        flow_aprx = signal.fftconvolve(cflw_aprx, cuih_aprx)[0:len(apfc_seas)]
        flow_aprx[flow_aprx < FLMN_CATC / INVL_SIMU] = FLMN_CATC / INVL_SIMU
        flow_catc = flow_aprx[np.arange(0, len(flow_aprx), APFC_CATC)]
        
        # Save attributes
        self.flow_catc = np.array(flow_catc, dtype = float)


class WaterProject(object):
    '''The characteristics and outcomes of the water project 
    
    Attributes: nmem_cwpr, fccp_cwpr, abav_cwpr, abst_cwpr, pfmb_cwpr, stor_cwpr, 
    wbal_cwpr, strs_cwpr, rslt_cwpr, stat_cwpr, memb_cwpr
    '''
    def __init__(self):
        
        # Set static and initial attributes
        self.nmem_cwpr = int(NMMN_CWPR * (np.random.beta(1 / (8 * NMCV_CWPR**2) - 0.5, 
                                                     1 / (8 * NMCV_CWPR**2) - 0.5, 
                                                     1)[0] 
                                          + 0.5))
        
        # Insert placeholders for attributes
        self.abav_cwpr = None
        self.abst_cwpr = None
        self.pfmb_cwpr = None
        self.stor_cwpr = None
        self.wbal_cwpr = None
        self.strs_cwpr = None
        self.rslt_cwpr = None
        self.stat_cwpr = None
        
        # Create members
        self.memb_cwpr = {}
        prob = np.random.random()        
        dcml = FCCP_CWPR * self.nmem_cwpr - int(FCCP_CWPR * self.nmem_cwpr)
        if prob <= dcml:
            ncop_cwpr = int(np.ceil(FCCP_CWPR * self.nmem_cwpr))
        else:
            ncop_cwpr = int(np.floor(FCCP_CWPR * self.nmem_cwpr))
        ndef_cwpr = int(self.nmem_cwpr - ncop_cwpr)
        smem_cwpr = np.append(np.zeros(ncop_cwpr), np.ones(ndef_cwpr))
        np.random.shuffle(smem_cwpr)
        for mbid_memb in range(self.nmem_cwpr):
            memb = Members()
            memb.mbid_memb = mbid_memb
            if smem_cwpr[mbid_memb] < 1:
                memb.stus_memb = 'cooperator'
                memb.smin_memb = SMNC_MEMB
                memb.smax_memb = SMXC_MEMB
            else:
                memb.stus_memb = 'defector'
                memb.smin_memb = SMND_MEMB
                memb.smax_memb = SMXD_MEMB
            self.memb_cwpr.update({mbid_memb:memb})
            
    def abav_calc(self, catc):
        '''Determine the water available for abstraction at each time step'''
        
        # Compare river flow to CWP capacity
        flow_cwpr = catc.flow_catc[(INVL_SIMU * BUFF_SIMU):] - FLMN_CWPR / INVL_SIMU
        hhrd_cwpr = np.array([memb.hhrd_memb for memb in self.memb_cwpr.values()])
        hhpr_cwpr = np.array([memb.hhpr_memb for memb in self.memb_cwpr.values()])
        tlpr_cwpr = np.bincount(hhrd_cwpr, hhpr_cwpr)
        abcc_cwpr = np.resize(np.repeat(tlpr_cwpr, INVL_SIMU) * PFMX_CWPR / INVL_SIMU, 
                              INVL_SIMU * TGRW_CROP)
        abav_cwpr = np.minimum(flow_cwpr, abcc_cwpr)
        abav_cwpr[abav_cwpr < 0] = 0.
        
        # Save attributes
        self.abav_cwpr = np.array(abav_cwpr, dtype = float)
    
    def memb_sims(self, loca):
        '''Simulate water allocation for CWP and members'''
        
        # Set initial values
        psto_invl = 0  # m**3
        abst_cwpr = []
        pfmb_cwpr = []
        stor_cwpr = []
        wbal_cwpr = []
        
        # Simulate member seasons
        for invl_tgrw in range(INVL_SIMU * TGRW_CROP):
            
            # Set storage and available flow
            stor_invl = psto_invl
            abav_invl = self.abav_cwpr[invl_tgrw]
            
            # Calculate member water demands
            for memb in self.memb_cwpr.values():
                memb.smst_step(self, loca, invl_tgrw)
            
            # Compare demand to available water and storage
            tldm_invl = np.sum([memb.tldm_memb[invl_tgrw] for memb in self.memb_cwpr.values()])
            abmb_invl = min(tldm_invl, abav_invl)  # m**3
            umdm_invl = max(0, round(tldm_invl - abmb_invl, 7))  # m**3
            stdm_invl = max(0, round(STOR_CWPR - stor_invl, 7))  # m**3
            abex_invl = max(0, round(abav_invl - abmb_invl, 7))  # m**3
            abso_invl = min(stdm_invl, abex_invl)  # m**3
            stmb_invl = min(umdm_invl, stor_invl)  # m**3
            abst_invl = abmb_invl + abso_invl  # m**3
            pfmb_invl = abmb_invl + stmb_invl
            
            # Calculate demand fraction
            if tldm_invl == 0:
                fcdm_invl = 0.
            else:
                fcdm_invl = pfmb_invl / tldm_invl
                
            # Update storage
            stor_invl = round(stor_invl + abso_invl - stmb_invl, 7)
        
            # Assign pipe flow to each hh and irrigate
            for memb in self.memb_cwpr.values():
                memb.pflw_memb[invl_tgrw] = fcdm_invl * memb.tldm_memb[invl_tgrw]
                memb.irrg_step(self, loca, invl_tgrw)
            
            # Save interval values
            abst_cwpr.append(abst_invl)
            pfmb_cwpr.append(pfmb_invl)
            stor_cwpr.append(stor_invl)
            
            # Update iterative values
            psto_invl = stor_invl
        
        # Calculate CWP water balance
        wbal_cwpr = np.round(np.array(abst_cwpr) - np.array(pfmb_cwpr) 
                             - np.diff(np.insert(np.array(stor_cwpr), 0, 0.)), 7)  # m**3
        
        # Save attributes
        self.abst_cwpr = np.array(abst_cwpr, dtype = float)
        self.stor_cwpr = np.array(stor_cwpr, dtype = float)
        self.pfmb_cwpr = np.array(pfmb_cwpr, dtype = float)
        self.wbal_cwpr = np.array(wbal_cwpr, dtype = float)
        
    def rslt_calc(self):
        '''Simulate seasonal results for CWP and members'''
    
        # Calculate seasonal CWP costs
        mcst_cwpr = CAMO_CWPR * self.nmem_cwpr * np.random.exponential(1., 1)
        wlim_cwpr = MBLM_CWPR * self.nmem_cwpr * TGRW_CROP
        if np.sum(self.abst_cwpr) <= wlim_cwpr:
            wcst_cwpr = np.sum(self.abst_cwpr) * COST_CWPR
        else:
            wcst_cwpr = COST_CWPR * (wlim_cwpr + (np.sum(self.abst_cwpr) - wlim_cwpr) * 1.5)
        self.rslt_cwpr = np.array([mcst_cwpr, wcst_cwpr, 0])
        
        # Calculate member results and aggregate totals
        for memb in self.memb_cwpr.values():
            memb.rslt_calc(self)
        strs_cwpr = np.array([memb.strs_memb for memb in self.memb_cwpr.values()])
        mbrs_cwpr = np.array([memb.rslt_memb for memb in self.memb_cwpr.values()])
        
        # Save attributes
        self.strs_cwpr = np.array(np.mean(strs_cwpr, 0), dtype = float)
        self.rslt_cwpr[2] = np.sum(mbrs_cwpr[:, 2])
        self.rslt_cwpr = np.array(self.rslt_cwpr, dtype = float)
        self.stat_cwpr = np.array([np.mean(mbrs_cwpr[:, 1]), 
                                   len(mbrs_cwpr[:, 1][mbrs_cwpr[:, 1] > CMIN_MEMB]) 
                                   / len(mbrs_cwpr[:, 1]), 
                                   np.sum(mbrs_cwpr[:, 2]) / (mcst_cwpr + wcst_cwpr)], 
                                  dtype = float)


class Locality(object):
    '''The characteristics and outcomes of the surrounding locality
    
    Attributes: nmem_loca, hhar_loca, kcrp_loca, prec_loca, smst_loca, 
    strs_loca, rslt_loca, stat_loca
    '''
    def __init__(self, cwpr):
        
        # Insert placeholders for attributes
        self.nmem_loca = np.array(cwpr.nmem_cwpr, dtype = float)
        self.hhar_loca = np.array([memb.hhar_memb for memb in cwpr.memb_cwpr.values()])
        self.kcrp_loca = np.array([memb.kcrp_memb for memb in cwpr.memb_cwpr.values()])
        self.prec_loca = None
        self.smst_loca = None
        self.strs_loca = None
        self.ntin_loca = None
        self.stat_loca = None
    
    def smst_sims(self, prec):
        '''Simulates soil moisture dynamics of the surrounding locality'''
        
        # Recalculate precipitation
        prec_loca = prec.rcrd_prec - ALDF_CLIM
        prec_loca[prec_loca < 0.] = 0.

        # Set initial values
        psms_invl = SFLD_SOIL
        smst_loca = []
        
        # Loop through intervals
        for prec_invl in prec_loca: 
            
            # Add Rainfall
            smst_invl = psms_invl + prec_invl / (PORO_SOIL * ZRLC_CROP)
            
            # Remove Surface Runoff
            if smst_invl > 1.:
                rnof_invl = smst_invl - 1.
                smst_invl = 1.
            else: 
                rnof_invl = 0.
            
            # Remove Leakage
            if smst_invl > SFLD_SOIL:
                leak_invl = (KSAT_SOIL / INVL_SIMU) \
                            * (np.exp(BETA_SOIL * (smst_invl - SFLD_SOIL)) - 1.) \
                            / (np.exp(BETA_SOIL * (1. - SFLD_SOIL)) - 1.)
                smst_invl = smst_invl - leak_invl / (PORO_SOIL * ZRLC_CROP)
            else:
                leak_invl = 0.
            
            # Remove Evapotranspiration Losses
            if smst_invl > SSTR_SOIL:
                evtr_invl = ETMX_CLIM / INVL_SIMU
            elif smst_invl > SWLT_SOIL and smst_invl <= SSTR_SOIL:
                evtr_invl = (EWLT_CLIM / INVL_SIMU) + ((ETMX_CLIM - EWLT_CLIM) / INVL_SIMU) \
                            * (smst_invl - SWLT_SOIL) / (SSTR_SOIL - SWLT_SOIL)
            elif smst_invl > SHYG_SOIL and smst_invl <= SWLT_SOIL:
                evtr_invl = (EWLT_CLIM / INVL_SIMU) * (smst_invl - SHYG_SOIL) \
                            / (SWLT_SOIL - SHYG_SOIL)
            else:
                evtr_invl = 0.
            smst_invl = smst_invl - evtr_invl / (PORO_SOIL * ZRLC_CROP)
            
            # Save interval Values
            smst_loca.append(smst_invl)
            
            # Update iterative values
            psms_invl = smst_invl
        
        # Save attributes
        self.prec_loca = np.array(prec_loca, dtype = float)
        self.smst_loca = np.array(smst_loca, dtype = float)
    
    def rslt_calc(self):
        '''Calculates average seasonal results for the surrounding locality'''
        
        # Calculate Average Static Stress
        sstr_loca = (SSTR_SOIL - self.smst_loca[(INVL_SIMU * BUFF_SIMU):]) \
                    / (SSTR_SOIL - SWLT_SOIL)  # dim
        sstr_loca = sstr_loca[sstr_loca > 0.]
        sstr_loca[sstr_loca > 1.] = 1.
        if len(sstr_loca) > 0:
            mstr_loca = np.mean(sstr_loca**QPAR_CROP)  # dim
        else:
            mstr_loca = 0.  # dim
        
        # Calculate Crossing Parameters
        indx_loca = np.where(self.smst_loca[(INVL_SIMU * BUFF_SIMU):] >= SSTR_SOIL)
        ccrs_loca = np.diff(np.append(0, np.append(indx_loca, INVL_SIMU 
                                                   * TGRW_CROP + 1))) - 1
        ccrs_loca = ccrs_loca[ccrs_loca > 0]
        ncrs_loca = len(ccrs_loca)  # dim
        if ncrs_loca > 0:
            mcrs_loca = np.mean(ccrs_loca) / INVL_SIMU  # days
        else:
            mcrs_loca = 0.
        
        # Calculate dynamic stress
        dstr_loca = ((mstr_loca * mcrs_loca) \
                    / (self.kcrp_loca * TGRW_CROP))**(ncrs_loca**-RPAR_CROP)
        dstr_loca[dstr_loca > 1.] = 1.
        
        # Calculate crop yields
        ycrp_loca = self.hhar_loca * YMAX_CROP * (1. - dstr_loca)  # kg
        
        # Calculate total income
        retr_loca = COST_CROP * ycrp_loca  # $
        ntin_loca = retr_loca
        
        # Save attributes
        self.ntin_loca = np.array(ntin_loca)
        self.strs_loca = np.array([np.mean(mstr_loca), np.mean(ncrs_loca), 
                                   np.mean(mcrs_loca), np.mean(dstr_loca)], 
                                  dtype = float)
        self.stat_loca = np.array([np.mean(ntin_loca), 
                                   len(ntin_loca[ntin_loca > CMIN_MEMB]) / len(ntin_loca), 
                                   np.nan], dtype = float)


class Members(object):
    '''The characteristics and outcomes of individual CWP members 
    
    Attributes: mbid_memb, stus_memb, smin_memb, smax_memb, hhar_memb, kcrp_memb,  
    hhpr_memb, hhst_memb, hhrd_memb, smst_memb, stor_memb, tldm_memb, pflw_memb, 
    irrg_memb, wbal_memb, strs_memb, rslt_memb
    '''
    
    def __init__(self):
        
        # Define static and initial attributes
        self.mbid_memb = None
        self.stus_memb = None
        self.smin_memb = None
        self.smax_memb = None
        self.hhar_memb = HAMN_MEMB * (np.random.beta(1 / (8 * HACV_MEMB**2) - 0.5, 
                                                     1 / (8 * HACV_MEMB**2) - 0.5, 
                                                     1)[0]
                                      + 0.5)  # m**2
        self.kcrp_memb = KPMN_CROP * (np.random.beta(1 / (8 * KPCV_CROP**2) - 0.5, 
                                                     1 / (8 * KPCV_CROP**2) - 0.5, 
                                                     1)[0]
                                      + 0.5)  # dim
        self.hhpr_memb = HPMN_MEMB * (np.random.beta(1 / (8 * HPCV_MEMB**2) - 0.5, 
                                                     1 / (8 * HPCV_MEMB**2) - 0.5, 
                                                     1)[0]
                                      + 0.5)  # dim
        self.hhst_memb = HSMN_MEMB * (np.random.beta(1 / (8 * HSCV_MEMB**2) - 0.5, 
                                                     1 / (8 * HSCV_MEMB**2) - 0.5, 
                                                     1)[0]
                                      + 0.5)  # m**3
        self.hhst_memb = np.round(self.hhst_memb, 2)
        self.hhrd_memb = int(np.floor(RDYS_CWPR * 
                                      np.random.randint(0, 10000, 1)[0] / 10000)) # dim
        
        # Insert placeholders for other attributes
        self.smst_memb = np.full(INVL_SIMU * TGRW_CROP, -1, dtype = float)
        self.stor_memb = np.full(INVL_SIMU * TGRW_CROP, -1, dtype = float)
        self.tldm_memb = np.full(INVL_SIMU * TGRW_CROP, -1, dtype = float)
        self.pflw_memb = np.full(INVL_SIMU * TGRW_CROP, -1, dtype = float)
        self.irrg_memb = np.full(INVL_SIMU * TGRW_CROP, -1, dtype = float)
        self.wbal_memb = None
        self.strs_memb = None
        self.rslt_memb = None
    
    def smst_step(self, cwpr, loca, invl_tgrw):
        '''Steps forward soil moisture dynamics and calculates irrigation demand'''
        
        # Set initial values
        if invl_tgrw < 1:
            psms_invl = loca.smst_loca[(INVL_SIMU * BUFF_SIMU - 1)]
            psto_invl = 0.  
        else:
            psms_invl = self.smst_memb[invl_tgrw - 1]
            psto_invl = self.stor_memb[invl_tgrw - 1]
        
        # Set storage and add precipitation
        stor_invl = psto_invl
        prec_invl = loca.prec_loca[(INVL_SIMU * BUFF_SIMU + invl_tgrw)]
            
        # Add Rainfall
        smst_invl = psms_invl + prec_invl / (PORO_SOIL * ZRLC_CROP)
            
        # Remove Surface Runoff
        if smst_invl > 1.:                
            rnof_invl = smst_invl - 1.
            smst_invl = 1.
        else: 
            rnof_invl = 0.
            
        # Remove Leakage
        if smst_invl > SFLD_SOIL:
            leak_invl = (KSAT_SOIL / INVL_SIMU) \
                        * (np.exp(BETA_SOIL * (smst_invl - SFLD_SOIL)) - 1.) \
                        / (np.exp(BETA_SOIL * (1. - SFLD_SOIL)) - 1.)
            smst_invl = smst_invl - leak_invl / (PORO_SOIL * ZRLC_CROP)
        else:
            leak_invl = 0.
            
        # Remove Evapotranspiration Losses
        if smst_invl > SSTR_SOIL:
            evtr_invl = ETMX_CLIM / INVL_SIMU
        elif smst_invl > SWLT_SOIL and smst_invl <= SSTR_SOIL:
            evtr_invl = (EWLT_CLIM / INVL_SIMU) + ((ETMX_CLIM - EWLT_CLIM) / INVL_SIMU) \
                        * (smst_invl - SWLT_SOIL) / (SSTR_SOIL - SWLT_SOIL)
        elif smst_invl > SHYG_SOIL and smst_invl <= SWLT_SOIL:
            evtr_invl = (EWLT_CLIM / INVL_SIMU) * (smst_invl - SHYG_SOIL) \
                        / (SWLT_SOIL - SHYG_SOIL)
        else:
            evtr_invl = 0.
        smst_invl = smst_invl - evtr_invl / (PORO_SOIL * ZRLC_CROP)
        
        # Calculate water demand
        if smst_invl < self.smin_memb:
            irdm_invl = (self.smax_memb - smst_invl) * (PORO_SOIL * ZRLC_CROP) * self.hhar_memb  # m**3
        else:
            irdm_invl = 0. * self.hhar_memb
        stdm_invl = max(0, round(self.hhst_memb - stor_invl, 7))  # m**3
        tldm_invl = irdm_invl + stdm_invl  # m**3
        
        # Amend demand based on max hh flow and rotation day
        if tldm_invl > self.hhpr_memb * PFMX_CWPR / INVL_SIMU:
            tldm_invl = self.hhpr_memb * PFMX_CWPR / INVL_SIMU
        if ((invl_tgrw // INVL_SIMU) % RDYS_CWPR) != self.hhrd_memb:
            tldm_invl = 0.
        
        # Save attributes
        self.smst_memb[invl_tgrw] = smst_invl
        self.stor_memb[invl_tgrw] = stor_invl
        self.tldm_memb[invl_tgrw] = tldm_invl
        
    def irrg_step(self, cwpr, loca, invl_tgrw):
        '''Simulates member irrigation and calculates water balance for one time step'''
        
        # Set initial values
        smst_invl = self.smst_memb[invl_tgrw]
        stor_invl = self.stor_memb[invl_tgrw]
        
        # Set pipe flow
        pflw_invl = self.pflw_memb[invl_tgrw]
        
        # Compare demand to available water and storage
        if smst_invl < self.smin_memb:
            irdm_invl = (self.smax_memb - smst_invl) * (PORO_SOIL * ZRLC_CROP) * self.hhar_memb  # m**3
        else:
            irdm_invl = 0. * self.hhar_memb
        pfir_invl = min(irdm_invl, pflw_invl)  # m**3
        umdm_invl = max(0, round(irdm_invl - pfir_invl, 7))  # m**3
        stdm_invl = max(0, round(self.hhst_memb - stor_invl, 7))  # m**3
        pfex_invl = max(0, round(pflw_invl - pfir_invl, 7))  # m**3
        pfso_invl = min(stdm_invl, pfex_invl)  # m**3
        stir_invl = min(umdm_invl, stor_invl)  # m**3
        irrg_invl = pfir_invl + stir_invl  # m**3
              
        # Update soil moisture and storage
        smst_invl = smst_invl + irrg_invl / self.hhar_memb \
                    / (PORO_SOIL * ZRLC_CROP) 
        stor_invl = round(stor_invl + pfso_invl - stir_invl, 7)
            
        # Update attributes
        self.smst_memb[invl_tgrw] = smst_invl
        self.stor_memb[invl_tgrw] = stor_invl
        self.irrg_memb[invl_tgrw] = irrg_invl
                    
    def rslt_calc(self, cwpr):
        '''Calculates seasonal results for each member'''
        
        # Calculate Water Balance
        wbal_memb = np.round(self.pflw_memb - self.irrg_memb 
                          - np.diff(np.insert(self.stor_memb, 0, 0.)), 7)  # m**3
        
        # Calculate Average Static Stress
        sstr_memb = (SSTR_SOIL - self.smst_memb) / (SSTR_SOIL - SWLT_SOIL)  # dim
        sstr_memb = sstr_memb[sstr_memb > 0.]
        sstr_memb[sstr_memb > 1.] = 1.
        if len(sstr_memb) > 0:
            mstr_memb = np.mean(sstr_memb**QPAR_CROP)  # dim
        else:
            mstr_memb = 0.  # dim
        
        # Calculate Crossing Parameters
        indx_memb = np.where(self.smst_memb >= SSTR_SOIL)
        ccrs_memb = np.diff(np.append(0, np.append(indx_memb, INVL_SIMU 
                                                   * TGRW_CROP + 1))) - 1
        ccrs_memb = ccrs_memb[ccrs_memb > 0]
        ncrs_memb = len(ccrs_memb)  # dim
        if ncrs_memb > 0:
            mcrs_memb = np.mean(ccrs_memb) / INVL_SIMU  # days
        else:
            mcrs_memb = 0.
        
        # Calculate dynamic stress
        dstr_memb = ((mstr_memb * mcrs_memb) \
                    / (self.kcrp_memb * TGRW_CROP))**(ncrs_memb**-RPAR_CROP)
        if dstr_memb > 1.:
            dstr_memb = 1.
        
        # Calculate crop yields
        ycrp_memb = self.hhar_memb * YMAX_CROP * (1. - dstr_memb)  # kg
        
        # Calculate total income
        retr_memb = COST_CROP * ycrp_memb  # $
        
        # Calculate amount paid to CWP and net return
        if BYVL_CWPR:
            if sum(cwpr.abst_cwpr) > 0:
                cfee_cwpr = (cwpr.rslt_cwpr[0] + cwpr.rslt_cwpr[1]) \
                        * sum(self.pflw_memb) / sum(cwpr.abst_cwpr)
            else:
                cfee_cwpr = cwpr.rslt_cwpr[0] / cwpr.nmem_cwpr
        else:
            cfee_cwpr = (cwpr.rslt_cwpr[0] + cwpr.rslt_cwpr[1]) / cwpr.nmem_cwpr
        if retr_memb > cfee_cwpr:
            cpad_memb = cfee_cwpr
            ntin_memb = retr_memb - cpad_memb
        else:
            cpad_memb = retr_memb
            ntin_memb = 0.
        
        # Save attributes
        self.wbal_memb = np.array(wbal_memb, dtype = float)
        self.strs_memb = np.array([mstr_memb, ncrs_memb, mcrs_memb, dstr_memb], 
                                  dtype = float)
        self.rslt_memb = np.array([retr_memb, ntin_memb, cpad_memb], dtype = float)
        

class Results(object):
    '''The simulation results to be saved for analysis
    
    Attributes: prec_rslt, loca_rslt, nnmb_rslt, flow_rslt, cwpr_rslt, memb_rslt, abfl_rslt
    '''
    
    def __init__(self):
        
        # Define Attributes
        self.prec_rslt = None
        self.loca_rslt = None
        self.nnmb_rslt = None
        self.flow_rslt = None
        self.cwpr_rslt = None
        self.memb_rslt = None
        self.abfl_rslt = None
   
    def rslt_clct(self, prec, catc, loca, cwpr):
        '''Collects results from all objects'''
        
        # Collect precipitation data
        prec_rslt = pd.DataFrame({'alph': pd.Series(prec.alph_prec, dtype = float), 
                                  'lmbd': pd.Series(prec.lmbd_prec, dtype = float), 
                                  'prdy': pd.Series(prec.prdy_prec, dtype = float)})
        
        # Collect nonmember locality data
        lctl_rslt = np.concatenate((np.array([loca.nmem_loca, 0]), loca.stat_loca)).reshape(1, -1)
        loca_rslt = pd.DataFrame(lctl_rslt, columns = ['nmem', 'abst', 'avin', 'fcsc', 'fcpd'], 
                                 dtype = float)
        loca_rslt['nmem'] = loca_rslt['nmem'].astype(int)
        
        # Collect individual nonmember data
        nnmb_rslt = pd.DataFrame({'nmid': pd.Series(np.arange(len(loca.hhar_loca)), 
                                                    dtype = int),
                                  'stus': pd.Series(['nonmember'] * len(loca.hhar_loca), 
                                                    dtype = str),
                                  'hhar': pd.Series(loca.hhar_loca, dtype = float), 
                                  'kcrp': pd.Series(loca.kcrp_loca, dtype = float),
                                  'hhpr': pd.Series(np.full(cwpr.nmem_cwpr, np.nan), 
                                                    dtype = float),
                                  'hhst': pd.Series(np.full(cwpr.nmem_cwpr, np.nan), 
                                                    dtype = float),
                                  'hhrd': pd.Series(np.full(cwpr.nmem_cwpr, np.nan), 
                                                    dtype = float), 
                                  'retr': pd.Series(loca.ntin_loca, dtype = float), 
                                  'ntin': pd.Series(loca.ntin_loca, dtype = float), 
                                  'cpad': pd.Series(np.full(cwpr.nmem_cwpr, 0), 
                                                    dtype = float)})
        
        # Collect flow data
        flow_rslt = pd.DataFrame({'invl': pd.Series(np.arange(INVL_SIMU * TGRW_CROP), 
                                                    dtype = int), 
                                  'flow': pd.Series(catc.flow_catc[(INVL_SIMU * BUFF_SIMU):], 
                                                    dtype = float)})
        
        # Collect CWP data
        cwtl_rslt = np.concatenate((np.array([cwpr.nmem_cwpr, np.sum(cwpr.abst_cwpr)]), 
                                    cwpr.stat_cwpr)).reshape(1, -1)
        cwpr_rslt = pd.DataFrame(cwtl_rslt, columns = ['nmem', 'abst', 'avin', 'fcsc', 'fcpd'], 
                                 dtype = float)
        cwpr_rslt['nmem'] = cwpr_rslt['nmem'].astype(int)
        
        # Collect individual member data
        memb_rslt = []
        for memb in cwpr.memb_cwpr.values():
            mbid_rslt = np.array([memb.mbid_memb, memb.stus_memb, memb.hhar_memb, 
                                  memb.kcrp_memb, memb.hhpr_memb, memb.hhst_memb,
                                  memb.hhrd_memb])
            mbtl_rslt = np.concatenate((mbid_rslt, memb.rslt_memb))
            memb_rslt.append(mbtl_rslt)
        memb_rslt = pd.DataFrame(memb_rslt, 
                                 columns = ['mbid', 'stus', 'hhar', 'kcrp', 'hhpr', 
                                            'hhst', 'hhrd', 'retr', 'ntin', 'cpad'], 
                                 dtype = float)
        memb_rslt['mbid'] = memb_rslt['mbid'].astype(int)
        memb_rslt['stus'] = memb_rslt['stus'].astype(str)
        memb_rslt['hhrd'] = memb_rslt['hhrd'].astype(int)
        
        # Collect abstraction data
        abfl_rslt = pd.DataFrame({'invl': pd.Series(np.arange(INVL_SIMU * TGRW_CROP), 
                                                    dtype = int), 
                                  'abfl': pd.Series(catc.flow_catc[(INVL_SIMU * BUFF_SIMU):] - 
                                                    cwpr.abst_cwpr, dtype = float)})
        
        # Save attributes
        self.prec_rslt = prec_rslt
        self.loca_rslt = loca_rslt
        self.nnmb_rslt = nnmb_rslt
        self.flow_rslt = flow_rslt
        self.cwpr_rslt = cwpr_rslt
        self.memb_rslt = memb_rslt
        self.abfl_rslt = abfl_rslt


# Define simulation function
def modl_simu(irun_simu):
    '''Run model and save results'''
    
    # Set new seed for each run
    np.random.seed(irun_simu)
    
    # Initialize objects
    prec = Precipitation()
    catc = Catchment()
    cwpr = WaterProject()
    loca = Locality(cwpr)
    rslt = Results()
    
    # Run model
    prec.prec_sims()
    catc.smst_sims(prec)
    catc.flow_sims()
    loca.smst_sims(prec)
    loca.rslt_calc()
    cwpr.abav_calc(catc)
    cwpr.memb_sims(loca)
    cwpr.rslt_calc()
    rslt.rslt_clct(prec, catc, loca, cwpr)
    
    # Add new columns with run and simulation number
    rslt.prec_rslt.insert(0, "irun", np.full(rslt.prec_rslt.shape[0], irun_simu), False)
    rslt.loca_rslt.insert(0, "irun", np.full(rslt.loca_rslt.shape[0], irun_simu), False)
    rslt.nnmb_rslt.insert(0, "irun", np.full(rslt.nnmb_rslt.shape[0], irun_simu), False)
    rslt.flow_rslt.insert(0, "irun", np.full(rslt.flow_rslt.shape[0], irun_simu), False)
    rslt.cwpr_rslt.insert(0, "irun", np.full(rslt.cwpr_rslt.shape[0], irun_simu), False)
    rslt.cwpr_rslt.insert(0, "simu", np.full(rslt.cwpr_rslt.shape[0], NMBR_SIMU), False)
    rslt.memb_rslt.insert(0, "irun", np.full(rslt.memb_rslt.shape[0], irun_simu), False)
    rslt.memb_rslt.insert(0, "simu", np.full(rslt.memb_rslt.shape[0], NMBR_SIMU), False)
    rslt.abfl_rslt.insert(0, "irun", np.full(rslt.abfl_rslt.shape[0], irun_simu), False)
    rslt.abfl_rslt.insert(0, "simu", np.full(rslt.abfl_rslt.shape[0], NMBR_SIMU), False)
    
    # Upload results tables to Sqlite database
    try:
        lock.acquire(True)
        
        # Static results
        if NMBR_SIMU == 0:
            rslt.prec_rslt.to_sql('prec_rslt', con = conn, if_exists = 'append', index = False)
            rslt.loca_rslt.to_sql('loca_rslt', con = conn, if_exists = 'append', index = False)
            rslt.nnmb_rslt.to_sql('nnmb_rslt', con = conn, if_exists = 'append', index = False)
            #rslt.flow_rslt.to_sql('flow_rslt', con = conn, if_exists = 'append', index = False)
        
        # Dynamic results
        rslt.cwpr_rslt.to_sql('cwpr_rslt', con = conn, if_exists = 'append', index = False)
        rslt.memb_rslt.to_sql('memb_rslt', con = conn, if_exists = 'append', index = False)
        #rslt.abfl_rslt.to_sql('abfl_rslt', con = conn, if_exists = 'append', index = False)
        
    finally:
        lock.release()


## Import parameters and run model
json_file = sys.argv[1]

with open(json_file, "r") as para_file:
    para_data = json.load(para_file)

cats = ['CLIM', 'CATC', 'SOIL', 'CROP', 'CWPR', 'MEMB', 'SIMU']
for catg in cats:
    for vble, valu in para_data[catg].items():
        try:
            exec(vble + ' = eval(valu)')
        except:
            exec(vble + ' = valu')
DBSE_FILE = para_data['DBSE_FILE']

# Create database and pool connections
conn = sqlite3.connect(DBSE_FILE)
lock = multiprocessing.Lock()
pool = multiprocessing.Pool(processes = multiprocessing.cpu_count())

# Collect simulation data and upload to database
simu_data = pd.DataFrame({'simu': pd.Series(NMBR_SIMU, dtype = int),
                          'rdys': pd.Series(RDYS_CWPR, dtype = float),
                          'cost': pd.Series(COST_CWPR, dtype = float),
                          'pfmx': pd.Series(PFMX_CWPR, dtype = float),
                          'byvl': pd.Series(BYVL_CWPR, dtype = str),
                          'fccp': pd.Series(FCCP_CWPR, dtype = float)})
simu_data.to_sql('simu_data', con = conn, if_exists = 'append', index = False)

# Print simulation number
print('Simulation:' + str(NMBR_SIMU))

# Run model
pool.map(modl_simu, range(0, NRUN_SIMU))

# Close connections
pool.close()
pool.join()
conn.close()