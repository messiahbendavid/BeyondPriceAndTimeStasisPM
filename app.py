# -*- coding: utf-8 -*-
"""
STASIS AM — Alpha Markets Server
Deploy to: stasisAM.beyondpriceandtime.com
Copyright © 2026 Truth Communications LLC. All Rights Reserved.

Requirements:
    pip install dash dash-bootstrap-components pandas numpy websocket-client requests gunicorn
"""

import time
import threading
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import copy
import json
import os

import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import websocket
import ssl
import requests

# ============================================================================
# CONFIG
# ============================================================================

POLYGON_API_KEY = os.environ.get(
    "POLYGON_API_KEY", "PnzhJOXEJO7tSpHr0ct2zjFKi6XO0yGi")


@dataclass
class Config:
    symbols: List[str] = field(default_factory=lambda: [
         "SPY", "AAPL", "NVDA", "TSLA", "AMD", "AMZN", "INTC", "PLTR", "BAC", "SOFI",
        "F", "GOOGL", "NIO", "MSFT", "META", "AAL", "SNAP", "CCL", "PFE", "T",
        "RIVN", "GOOG", "MARA", "UBER", "COIN", "VZ", "WFC", "NFLX", "KO", "LCID",
        "RIOT", "PYPL", "KVUE", "DIS", "VALE", "BABA", "CMCSA", "CSCO", "GM", "MU",
        "HOOD", "PARA", "MPW", "C", "CLSK", "JPM", "JNJ", "DAL", "WBA", "GRAB",
        "PBR", "GOLD", "AGNC", "AVGO", "JBLU", "XOM", "CVX", "SMCI", "KMI", "SQ",
        "SIRI", "BMY", "ET", "PCG", "TSM", "SCHW", "ABR", "PLUG", "DKNG", "HBAN",
        "LUV", "WBD", "NCLH", "RIG", "SHOP", "UAL", "DVN", "MS", "KEY", "ORCL",
        "OXY", "VFC", "ROKU", "TFC", "AFRM", "ARM", "HAL", "MO", "CLF", "WMT",
        "CHPT", "FCX", "PINS", "UPST", "GILD", "ABBV", "USB", "NU", "PDD", "QS",
        "SE", "ENVX", "RF", "JD", "SAVE", "SLB", "BRK.B", "CRM", "MRK", "NKLA",
        "PENN", "GS", "LI", "MGM", "UNH", "NEM", "BX", "XPEV", "ABNB", "BA",
        "COP", "AG", "WYNN", "ENPH", "AXP", "PM", "FSLR", "V", "MA", "LRCX",
        "COST", "UPS", "OPEN", "SBUX", "PANW", "ADBE", "MRVL", "ON", "IONQ", "AEM",
        "CRWD", "NOW", "VRT", "RBLX", "SPCE", "PATH", "TXN", "NKE", "DOW", "TELL",
        "QCOM", "LOW", "SNOW", "HD", "ANET", "SEDG", "BTU", "CAT", "DG", "MMM",
        "DASH", "MPC", "PSX", "FITB", "CZR", "BP", "LYFT", "LVS", "ASML", "SPOT",
        "EBAY", "MDT", "CVS", "RTX", "MTCH", "RKLB", "LLY", "HON", "SPWR", "APA",
        "EOSE", "BTI", "ZIM", "AFL", "EQT", "VLO", "BKR", "MRNA", "Z", "MRO",
        "DE", "LUMN", "AIG", "TEVA", "ZS", "CF", "TJX", "SWN", "CELH", "CMG",
        "TTWO", "AR", "U", "GE", "DHR", "STLA", "CL", "ALLY", "APD", "WPM",
        "BILI", "STX", "ETSY", "CPNG", "IMGN", "LAZR", "PG", "SNDL", "TTD", "NET",
        "AMC", "ADM", "DDOG", "MDB", "WDAY", "DELL", "STNG", "TWLO", "OKTA", "DOCU",
        "SU", "GSAT", "ZION", "FUBO", "CHWY", "WDC", "BLK", "TLRY", "CNC", "PEP",
        "BIDU", "ZM", "EL", "CARR", "FDX", "UEC", "ASTS", "CIG", "HPE", "CNP",
        "SYF", "ILMN", "TMO", "AZN", "AMGN", "PXD", "REGN", "ISRG", "RCL", "LEN",
        "TAL", "GPN", "D", "CTRA", "SO", "NEE", "DUK", "AEP", "SRE", "EXC",
        "VICI", "SPG", "O", "AMT", "CCI", "PLD", "WELL", "DLR", "EQIX", "PSA",
        "SBAC", "ARE", "AVB", "ARES", "KKR", "APO", "GDDY", "VEEV", "HPQ", "WHR",
        "CNX", "EDR", "NYCB", "COTY", "RUN", "GIS", "CPB", "HST", "IQ", "IP",
        "TRMB", "ACB", "CGC", "CRON", "TLRY", "DNA", "JOBY", "CIFR", "IREN", "HUT",
        "BITF", "BTBT", "CORZ", "WULF", "IBRX", "SOUN", "AI", "BBAI", "BIGC", "BLNK",
        "GOEV", "EVGO", "CHARX", "FSR", "MULN", "FFIE", "PSNY", "PTRA", "REE", "RIDE",
        "ARVL", "WKHS", "HYLN", "XL", "VTOL", "MP", "LAC", "ALB", "LTHM", "SLI",
        "PLL", "SGML", "ALTM", "SQM", "STNE", "PAGS", "DLO", "MELI", "GLOB", "VTEX",
        "CRSR", "LOGI", "HEAR", "GPRO", "SONO", "AAON", "ABNB", "ACGL", "ACM", "ACN",
        "ADSK", "AEE", "AES", "ALGN", "ALLE", "AMAT", "AMBA", "AMCR", "AME", "AMKR",
        "AMP", "ANSS", "AON", "APH", "APTV", "ARCC", "ARWR", "ATHM", "ATVI", "AUY",
        "AXON", "AZO", "BALL", "BAX", "BBWI", "BDX", "BEN", "BERY", "BFAM", "BIIB",
        "BIO", "BJ", "BK", "BKNG", "BKSY", "BKU", "BNTX", "BOX", "BR", "BRO",
        "BSX", "BWA", "BXP", "BYND", "CAKE", "CALM", "CANO", "CB", "CBRE", "CCEP",
        "CDAY", "CDNS", "CDW", "CE", "CEG", "CERN", "CHD", "CHDN", "CHK", "CHRW",
        "CI", "CINF", "CLVS", "CME", "CMI", "CMS", "CMA", "CNQ", "COF", "COHR",
        "COMM", "COOP", "COUR", "CPE", "CPRI", "CPT", "CRL", "CRUS", "CROX", "CTLT",
        "CTSH", "CTVA", "CVNA", "CW", "CWH", "CXO", "CYBR", "CYTK", "DAR", "DAVA",
        "DBX", "DCI", "DD", "DDOG", "DECK", "DFS", "DHI", "DINO", "DISCA", "DISH",
        "DLTR", "DNLI", "DOCS", "DOOO", "DPZ", "DRI", "DT", "DTE", "DUOL", "DV",
        "DVAX", "DXC", "DXCM", "EA", "ECL", "ED", "EFX", "EIX", "EL", "EME",
        "EMN", "EMR", "ENDP", "ENOV", "ENTG", "EOG", "EPAM", "EQR", "EQT", "ERJ",
        "ES", "ESS", "ESTC", "ETN", "ETR", "ETRN", "ETSY", "EVR", "EVRG", "EW",
        "EXAS", "EXPD", "EXPE", "EXR", "FANG", "FAST", "FBIN", "FBRT", "FDS", "FDUS",
        "FE", "FERG", "FFIV", "FHN", "FIS", "FISV", "FIVE", "FLR", "FLT", "FMC",
        "FND", "FNF", "FOCS", "FOX", "FOXA", "FRPT", "FSLY", "FSM", "FSTR", "FTAI",
        "FTCH", "FTNT", "FTV", "FVRR", "G", "GATX", "GBCI", "GD", "GDYN", "GEL",
        "GEN", "GENI", "GFL", "GFS", "GH", "GIB", "GL", "GLBE", "GLOB", "GLPI",
        "GLW", "GMED", "GNRC", "GPC", "GPI", "GRMN", "GRUB", "GRWG", "GSHD", "GT",
        "GTLB", "GTLS", "GWW", "HAS", "HASI", "HBAN", "HBI", "HCA", "HCAT", "HCP",
        "HELE", "HES", "HIG", "HII", "HL", "HLF", "HLT", "HOLX", "HOV", "HRB",
        "HRL", "HSBC", "HSIC", "HSY", "HTHT", "HUBB", "HUBS", "HUM", "HWM", "HXL",
        "IAC", "ICE", "IDXX", "IEX", "IIVI", "INCY", "INFA", "INFY", "ING", "INMD",
        "INSM", "INSP", "INST", "INTC", "INTU", "INVH", "IONS", "IOVA", "IPG", "IQV",
        "IR", "IRBT", "IRM", "IRTC", "IT", "ITCI", "ITW", "IVZ", "JACK", "JBHT",
        "JCI", "JKHY", "JMIA", "JWN", "K", "KBH", "KD", "KDP", "KEX", "KEYS",
        "KGC", "KHC", "KIM", "KLAC", "KMB", "KMX", "KNX", "KOS", "KR", "KSS",
        "L", "LAUR", "LAZR", "LBRDK", "LDOS", "LEA", "LECO", "LEG", "LEVI", "LGIH",
        "LHCG", "LHX", "LIN", "LITE", "LKQ", "LLAP", "LNC", "LNG", "LNT", "LNTH",
        "LPLA", "LPSN", "LSCC", "LSXMA", "LULU", "LW", "LYB", "LYV", "M", "MAA",
        "MAN", "MANH", "MAR", "MAS", "MASI", "MATX", "MCHP", "MCK", "MCO", "MDGL",
        "MDLZ", "MDRX", "MEG", "MET", "MFST", "MGNI", "MGNX", "MHK", "MKTX", "MLM",
        "MNDY", "MNST", "MOD", "MOH", "MOS", "MPWR", "MRCY", "MRKR", "MRNA", "MRVI",
        "MSGS", "MSTR", "MTDR", "MTEM", "MTG", "MTH", "MTN", "MTSI", "MTZ", "MUSA",
        "NARI", "NBIX", "NDAQ", "NDSN", "NEE", "NEM", "NEP", "NEWR", "NFE", "NI",
        "NLY", "NMIH", "NOC", "NOG", "NOV", "NRG", "NSC", "NTAP", "NTES", "NTNX",
        "NTRS", "NUAN", "NUE", "NVR", "NVS", "NWSA", "NWS", "NXP", "NXPI", "NXST",
        "NYT", "OC", "ODFL", "OGN", "OGS", "OHI", "OKE", "OLED", "OLLI", "OMC",
        "ONON", "OPK", "ORA", "ORLY", "OSCR", "OSH", "OTIS", "OUT", "PCAR", "PCH",
        "PCOR", "PCTY", "PEAK", "PEG", "PERI", "PFG", "PGNY", "PGR", "PH", "PHG",
        "PHM", "PKG", "PKI", "PKX", "PLTK", "PM", "PNC", "PNR", "PNW", "POOL",
        "POST", "POWW", "PPG", "PPL", "PRCT", "PRGO", "PRI", "PRLB", "PRMW", "PRU",
        "PSTG", "PTC", "PTON", "PVH", "PWR", "PXD", "PYPL", "QGEN", "QRVO", "QUOT",
        "R", "RBA", "RBLX", "RCM", "RDFN", "RE", "REAL", "REG", "RERE", "REVG",
        "REXR", "REZI", "RH", "RHI", "RHP", "RJF", "RL", "RMBS", "RMD", "RNG",
        "RNST", "ROK", "ROL", "ROP", "ROST", "RPD", "RPM", "RPRX", "RS", "RSI",
        "RTO", "RTRX", "RVLV", "RXO", "RYAN", "RYAAY", "S", "SAH", "SAIA", "SAM",
        "SAP", "SBGI", "SBS", "SBSW", "SCCO", "SCHW", "SCI", "SEDG", "SEE", "SHAK",
        "SHEL", "SHW", "SIGI", "SJM", "SKIL", "SKX", "SKYH", "SLGN", "SM", "SNA",
        "SNDR", "SNPS", "SNV", "SNX", "SO", "SOLV", "SON", "SOS", "SPB", "SPHB",
        "SPI", "SPLK", "SPOT", "SPR", "SPTN", "SRPT", "SSP", "SSNC", "ST", "STAG",
        "STAA", "STLD", "STN", "STRA", "STRL", "STT", "STZ", "SUI", "SUM", "SUN",
        "SUPN", "SWAV", "SWI", "SWK", "SWKS", "SYK", "SYM", "SYNA", "SYY", "TAP",
        "TCOM", "TCPC", "TD", "TDG", "TDY", "TEAM", "TECH", "TEN", "TER", "TGNA",
        "TGT", "THS", "TKC", "TKO", "TLK", "TME", "TMHC", "TMUS", "TNC", "TNDM",
        "TNET", "TOL", "TOST", "TPG", "TPR", "TREX", "TRGP", "TRIP", "TRMB", "TROW",
        "TRP", "TRV", "TSN", "TT", "TTC", "TTEC", "TU", "TUYA", "TVTX", "TW",
        "TWKS", "TWST", "TXG", "TXRH", "TYL", "UDR", "UFPI", "UGI", "UHS", "UI",
        "ULTA", "UMC", "UNM", "UNP", "UPWK", "URI", "URBN", "USFD", "UTHR", "UWMC",
        "VAIL", "VEON", "VERX", "VFC", "VIPS", "VIRT", "VKTX", "VLY", "VMC", "VNDA",
        "VNO", "VNOM", "VRSK", "VRSN", "VRTS", "VRTX", "VST", "VTRS", "VTR", "VXRT",
        "W", "WAB", "WAL", "WAT", "WBS", "WCC", "WCN", "WDAY", "WEC", "WELL",
        "WEN", "WERN", "WES", "WEX", "WFRD", "WH", "WIX", "WLK", "WOLF", "WPC",
        "WPP", "WRB", "WRK", "WSC", "WSM", "WSO", "WST", "WTFC", "WTM", "WTS",
        "WU", "WW", "WWD", "X", "XEL", "XP", "XPEL", "XPO", "XRAY", "XRX",
        "XYL", "Y", "YETI", "YMM", "YPF", "YUM", "YUMC", "ZBH", "ZBRA", "ZEN",
        "ZG", "ZGN", "ZI", "ZLAB", "ZNH", "ZNTL", "ZTS", "ZUO", "ZWS", "ACIW",
        "ACLS", "ACLX", "ACVA", "ADAP", "ADI", "ADNT", "ADP", "ADPT", "ADTN", "ADVM",
        "AEG", "AEIS", "AER", "AFYA", "AGCO", "AGFS", "AGIO", "AGTI", "AHCO", "AHH",
        "AHR", "AIT", "AJRD", "AKAM", "AKR", "ALEC", "ALGM", "ALIT", "ALK", "ALKS",
        "ALKT", "ALL", "ALNY", "ALTG", "ALTR", "ALV", "ALVR", "AMAT", "AMBA", "AMCX",
        "AMEH", "AMG", "AMGN", "AMH", "AMKR", "AMP", "AMPH", "AMR", "AMRC", "AMRX",
        "AMSC", "AMWD", "ANAB", "ANDE", "ANET", "ANGI", "ANIP", "ANSS", "ANTE", "AOS",
        "AORT", "AP", "APAM", "APG", "APGE", "API", "APLS", "APLT", "APO", "APPF",
        "APPN", "APPS", "AQN", "AQST", "ARCB", "ARCC", "ARCO", "ARCT", "ARDX", "ARHS",
        "ARI", "ARIS", "ARKO", "ARLO", "AROC", "ARVN", "ARWR", "ASGN", "ASIX", "ASO",
        "ASPN", "ASTE", "ASUR", "ATKR", "ATMU", "ATRA", "ATRC", "ATSG", "AVAV", "AVDX",
        "AVID", "AVNT", "AVPT", "AVTR", "AVXL", "AWI", "AWK", "AXNX", "AXSM", "AYI",
        "AZEK", "AZZ", "BAND", "BANF", "BANR", "BASE", "BBIO", "BBSI", "BCPC", "BDC",
        "BDTX", "BE", "BEAM", "BECN", "BFH", "BGCP", "BGNE", "BHC", "BHVN", "BJ",
        "BJRI", "BKE", "BKH", "BKRIF", "BKU", "BLBD", "BLD", "BLDR", "BLFS", "BLI",
        "BLMN", "BLTE", "BLUE", "BMBL", "BMI", "BMRN", "BNRE", "BOH", "BOOT", "BORR",
        "BR", "BRCC", "BRBR", "BRC", "BRFS", "BRO", "BROS", "BRSP", "BRZE", "BSM",
        "BSX", "BURL", "BWXT", "BXC", "BXP", "BXSL", "BYD", "BYSI", "CAG", "CALA",
        "CALM", "CAMT", "CARG", "CARS", "CASY", "CATY", "CAVA", "CBAN", "CBOE", "CBRE",
        "CBT", "CBU", "CBZ", "CCCS", "CCOI", "CCRN", "CCS", "CCSI", "CDAY", "CDLX",
        "CDNA", "CDNS", "CDW", "CE", "CEIX", "CENX", "CERN", "CEVA", "CFG", "CFLT",
        "CFR", "CGNX", "CHCT", "CHD", "CHDN", "CHE", "CHH", "CHK", "CHNG", "CHRW",
        "CHS", "CHTR", "CHUY", "CI", "CIEN", "CIM", "CINF", "CIR", "CIT", "CIVB",
        "CIVI", "CLAR", "CLB", "CLBT", "CLDT", "CLDX", "CLH", "CLI", "CLNE", "CLNX",
        "CLSK", "CLVR", "CLVT", "CLW", "CMC", "CME", "CMI", "CMP", "CMPO", "CMPR",
        "CMS", "CNA", "CNHI", "CNK", "CNM", "CNMD", "CNNE", "CNO", "CNSL", "CNXC",
        "CNXN", "COCO", "CODI", "COHR", "COLB", "COLD", "COLL", "COMM", "COMP", "COOP",
        "CORT", "COTY", "COUR", "CP", "CPE", "CPF", "CPNG", "CPRI", "CPT", "CRC",
        "CREE", "CRI", "CRK", "CRL", "CRMT", "CRNC", "CRNT", "CROX", "CRS", "CRSP",
        "CRVL", "CRVS", "CRWD", "CSGP", "CSL", "CSOD", "CSTL", "CSTM", "CSWI", "CSX",
        "CTAS", "CTLP", "CTLT", "CTO", "CTSH", "CTVA", "CUBE", "CUK", "CURO", "CVAC",
        "CVBF", "CVC", "CVCO", "CVGW", "CVI", "CVLT", "CVNA", "CW", "CWST", "CXM",
        "CXW", "CYBR", "CYRX", "CYTK", "CZMR", "DADA", "DAR", "DASH", "DAVA", "DAY",
        "DBD", "DBI", "DBX", "DCGO", "DCI", "DCPH", "DD", "DDL", "DDOG", "DECA",
        "DECK", "DEI", "DELL", "DENN", "DEO", "DERM", "DESP", "DFH", "DFS", "DGII",
        "DHC", "DHI", "DHT", "DIBS", "DINO", "DIOD", "DISH", "DKL", "DKS", "DLB",
        "DLTR", "DLX", "DMRC", "DNLI", "DNMR", "DNR", "DO", "DOC", "DOCS", "DOCN",
        "DOOR", "DORM", "DOV", "DPZ", "DQ", "DRCT", "DRI", "DRVN", "DSEY", "DSGN",
        "DSGX", "DSKE", "DSP", "DT", "DTE", "DTIL", "DUOL", "DV", "DVA", "DVAX",
        "DXC", "DXCM", "DY", "DYN", "DYNT", "E", "EA", "EAF", "EAT", "EATZ",
        "EB", "EBC", "EBET", "EBR", "ECPG", "ECVT", "ED", "EDIT", "EDR", "EFC",
        "EFX", "EGHT", "EGO", "EGP", "EHC", "EIX", "ELAN", "ELF", "ELV", "ELVN",
        "EME", "EMN", "EMR", "ENB", "ENDO", "ENOV", "ENR", "ENS", "ENSG", "ENTA",
        "ENTG", "ENVX", "EOG", "EPAM", "EPD", "EPR", "EQH", "EQR", "EQRX", "EQST",
        "ERA", "ERES", "ERF", "ERII", "ERJ", "ERO", "ES", "ESAB", "ESMT", "ESNT",
        "ESRT", "ESS", "ESTA", "ESTC", "ETFC", "ETN", "ETR", "ETRN", "EVBG", "EVC",
        "EVER", "EVH", "EVO", "EVOP", "EVRI", "EVRG", "EW", "EWBC", "EXAS", "EXPD",
        "EXPE", "EXPO", "EXR", "EXTR", "EYEN", "EZPW", "FAF", "FANG", "FAST", "FATE",
        "FBIN", "FBMS", "FBNC", "FBRT", "FCFS", "FCHI", "FCNCA", "FCX", "FDS", "FDUS",
        "FE", "FELE", "FERG", "FG", "FGEN", "FHB", "FHI", "FHN", "FI", "FIBK",
        "FICO", "FIS", "FISV", "FIX", "FIZZ", "FL", "FLDM", "FLEX", "FLGT", "FLIC",
        "FLNC", "FLNG", "FLO", "FLOW", "FLR", "FLS", "FLT", "FLUX", "FMC", "FMS",
        "FN", "FNA", "FNB", "FND", "FNF", "FNKO", "FOCS", "FOLD", "FORM", "FORR",
        "FOX", "FOXA", "FOXF", "FPH", "FPRX", "FR", "FREE", "FRGE", "FRME", "FRO",
        "FRPT", "FRSH", "FRST", "FSBC", "FSLR", "FSLY", "FSM", "FSS", "FSTR", "FTAI",
        "FTCH", "FTDR", "FTNT", "FTRE", "FTS", "FTV", "FUTU", "FVRR", "FVR", "FWRD",
        "FWRG", "G", "GATX", "GBCI", "GBDC", "GBT", "GBX", "GCO", "GD", "GDYN",
        "GEF", "GEL", "GEN", "GENI", "GERN", "GES", "GEVO", "GFF", "GFL", "GFS",
        "GH", "GHC", "GHRS", "GIB", "GIII", "GIL", "GILD", "GKOS", "GL", "GLBE",
        "GLDD", "GLEO", "GLNG", "GLOB", "GLOP", "GLPI", "GLW", "GMED", "GMRE", "GMS",
        "GNRC", "GNW", "GO", "GOCO", "GOLF", "GOOD", "GOOS", "GPC", "GPI", "GPMT",
        "GPOR", "GPRE", "GPS", "GRBK", "GREE", "GREK", "GRI", "GRIN", "GRMN", "GRND",
        "GROV", "GROW", "GS", "GSHD", "GSIT", "GT", "GTES", "GTLB", "GTLS", "GTN",
        "GTY", "GURL", "GVA", "GWW", "GXO", "H", "HAE", "HAFC", "HALL", "HAS",
        "HAYN", "HBI", "HCA", "HCAT", "HCC", "HCI", "HCKT", "HD", "HDB", "HEAR",
        "HEES", "HEI", "HELE", "HES", "HESM", "HGV", "HIG", "HIBB", "HII", "HIMX",
        "HIMS", "HL", "HLF", "HLI", "HLLY", "HLNE", "HLT", "HLVX", "HMN", "HOLI",
        "HOLX", "HON", "HOPE", "HOV", "HP", "HQY", "HR", "HRB", "HRI", "HRL",
        "HRMY", "HROW", "HSBC", "HSIC", "HST", "HSY", "HTHT", "HTLF", "HUBG", "HUBB",
        "HUBS", "HUM", "HURN", "HVT", "HWC", "HWM", "HXL", "HY", "HYFM", "HZNP",
        "IAC", "IART", "IBEX", "IBKR", "IBN", "IBOC", "IBP", "ICE", "ICL", "ICUI",
        "IDCC", "IDEX", "IDT", "IDXX", "IEP", "IEX", "IFNNY", "IGT", "IHRT", "IHS",
        "III", "IIIN", "IIIV", "IIVI", "IMMR", "IMMU", "IMVT", "INBK", "INCY", "INDB",
        "INDI", "INFA", "INFN", "INFY", "ING", "INGN", "INGR", "INMD", "INN", "INOD",
        "INSM", "INSP", "INST", "INTU", "INVH", "IOSP", "IOVA", "IPG", "IPGP", "IQ",
        "IQV", "IR", "IRBT", "IRDM", "IRET", "IRM", "IRTC", "ISBC", "ISEE", "IT",
        "ITCI", "ITGR", "ITT", "ITW", "IVZ", "JACK", "JAMF", "JANX", "JAX", "JAZZ",
        "JBGS", "JBHT", "JBL", "JBSS", "JBT", "JCI", "JELD", "JEF", "JFIN", "JHG",
        "JJSF", "JKHY", "JLL", "JMIA", "JNPR", "JOBY", "JOE", "JWN", "JYNT", "K",
        "KAI", "KALU", "KAR", "KB", "KBH", "KBR", "KD", "KDP", "KE", "KELYA",
        "KEN", "KEX", "KEYS", "KFY", "KGC", "KHC", "KIM", "KIN", "KLIC", "KLAC",
        "KMB", "KMT", "KMX", "KN", "KNX", "KOD", "KNTK", "KOS", "KPTI", "KR",
        "KRC", "KREF", "KRG", "KRNY", "KROS", "KRP", "KSS", "KTB", "KTOS", "KURA",
        "KVHI", "KW", "KWR", "KYMR", "L", "LAD", "LANC", "LAND", "LAR", "LARK",
        "LAUR", "LAW", "LBPH", "LBRDA", "LBRDK", "LCII", "LCUT", "LDOS", "LEA", "LECO",
        "LEG", "LEGN", "LESL", "LEVI", "LGF.A", "LGIH", "LGND", "LH", "LHX", "LHCG",
        "LI", "LILA", "LILAK", "LIND", "LINE", "LINTA", "LION", "LITE", "LIVN", "LKFN",
        "LL", "LLAP", "LMND", "LMNR", "LMT", "LNC", "LNG", "LNKD", "LNN", "LNT",
        "LNTH", "LNVGY", "LOB", "LOCO", "LOGI", "LOOP", "LOVE", "LPLA", "LPX", "LRCX",
        "LRMR", "LSCC", "LSI", "LSPD", "LSTR", "LSXMA", "LTCH", "LTHM", "LTRN", "LULU",
        "LUNR", "LUV", "LVIS", "LVS", "LW", "LXRX", "LYB", "LYEL", "LYG", "LYRA",
        "LYV", "LZB", "M", "MA", "MAA", "MAC", "MACK", "MAESY", "MAN", "MANH",
        "MANT", "MAR", "MAS", "MASI", "MAT", "MATV", "MATX", "MAX", "MBIN", "MBI",
        "MC", "MCB", "MCBS", "MCK", "MCO", "MCRI", "MCW", "MCY", "MD", "MDGL",
        "MDLZ", "MDRX", "MDU", "MEDP", "MEG", "MEI", "MELI", "MEN", "MEOH", "MET",
        "MFA", "MFG", "MFIN", "MGA", "MGEE", "MGI", "MGIC", "MGLN", "MGNI", "MGNX",
        "MGPI", "MGY", "MHH", "MHK", "MHO", "MIC", "MIDD", "MIME", "MIND", "MIRM",
        "MISO", "MITK", "MITT", "MKSI", "MKTX", "MLCO", "MLI", "MLM", "MLNK", "MMIT",
        "MMS", "MMSI", "MNKD", "MNDY", "MNR", "MNRL", "MNST", "MOD", "MODG", "MODN",
        "MODV", "MOG.A", "MOH", "MOR", "MOS", "MOV", "MPAA", "MPWR", "MQ", "MRCY",
        "MRIN", "MRK", "MRNA", "MRNS", "MRSN", "MRV", "MRVI", "MRVL", "MSGS", "MSI",
        "MSM", "MSPR", "MSTR", "MT", "MTB", "MTCH", "MTDR", "MTEK", "MTEX", "MTG",
        "MTH", "MTN", "MTOR", "MTRN", "MTSI", "MTTR", "MTX", "MTZ", "MUR", "MUSA",
        "MVST", "MWA", "MXL", "MYOV", "MYPS", "MYRG", "NAD", "NARI", "NAT", "NATH",
        "NBHC", "NBI", "NBIX", "NBNK", "NBR", "NC", "NCBS", "NCMI", "NCNO", "NCSM",
        "NDAQ", "NDLS", "NDSN", "NEE", "NEM", "NEO", "NEOG", "NEP", "NET", "NEWR",
        "NEXT", "NFBK", "NFE", "NFG", "NFGC", "NHI", "NI", "NICE", "NJR", "NKE",
        "NKLA", "NKTR", "NLY", "NMFC", "NMIH", "NMRK", "NNN", "NNOX", "NOA", "NOAC",
        "NOC", "NOG", "NOR", "NOV", "NOVA", "NOVT", "NOW", "NP", "NPO", "NR",
        "NRDS", "NRG", "NRIX", "NSA", "NSC", "NSIT", "NSP", "NTAP", "NTCT", "NTES",
        "NTLA", "NTNX", "NTR", "NTRS", "NTST", "NTUS", "NUAN", "NUE", "NUVB", "NVG",
        "NVGS", "NVO", "NVR", "NVS", "NVST", "NVT", "NWBI", "NWE", "NWL", "NWPX",
        "NWS", "NWSA", "NX", "NXE", "NXPI", "NXRT", "NXST", "NYCB", "NYT", "OAS",
        "OC", "OCFC", "OCFT", "OCGN", "ODFL", "ODP", "OEC", "OFG", "OFIX", "OGE",
        "OGN", "OGS", "OHI", "OI", "OII", "OIS", "OKE", "OLED", "OLLI", "OLN",
        "OLP", "OM", "OMC", "OMCL", "OMF", "OMI", "ONB", "ONEW", "ONTO", "ONON",
        "OPAD", "OPI", "OPK", "OPRA", "OPRX", "ORA", "ORCL", "ORI", "ORLA", "ORLY",
        "OSH", "OSIS", "OSK", "OSPN", "OSTK", "OTIS", "OUST", "OUT", "OVV", "OWL",
        "OXLC", "OXM", "OZK", "PAA", "PAAS", "PAC", "PACB", "PACK", "PAG", "PAGP",
        "PAGS", "PAHC", "PANW", "PAR", "PARR", "PATH", "PAYO", "PAYC", "PAYX", "PB",
        "PBA", "PBCT", "PBF", "PBH", "PBYI", "PCAR", "PCG", "PCH", "PCOR", "PCRX",
        "PCTY", "PCVX", "PDCE", "PDCO", "PDD", "PDFS", "PEAK", "PEB", "PECO", "PEG",
        "PEGA", "PEN", "PENN", "PEP", "PERI", "PETQ", "PETS", "PFG", "PFGC", "PFSI",
        "PG", "PGR", "PGNY", "PGY", "PH", "PHG", "PHM", "PHR", "PHVS", "PI",
        "PIII", "PINC", "PINE", "PING", "PINS", "PIPR", "PIRS", "PKBK", "PKG", "PKI",
        "PKX", "PL", "PLAN", "PLAY", "PLBY", "PLD", "PLNT", "PLRX", "PLSE", "PLTK",
        "PLTR", "PLXS", "PLYA", "PM", "PMT", "PMVP", "PNC", "PNFP", "PNM", "PNR",
        "PNW", "PODD", "POOL", "POR", "POST", "POWL", "POWW", "PPG", "PPL", "PR",
        "PRA", "PRAA", "PRCH", "PRCT", "PRDO", "PRFT", "PRGO", "PRGS", "PRI", "PRIM",
        "PRLB", "PRMW", "PRO", "PROC", "PROG", "PROV", "PRPL", "PRTA", "PRTK", "PRU",
        "PRVB", "PS", "PSA", "PSB", "PSEC", "PSTG", "PSTH", "PSX", "PTEN", "PTC",
        "PTCT", "PTE", "PTGX", "PTON", "PTVE", "PVH", "PWR", "PXD", "PYCR", "PYPL",
        "PZN", "QCOM", "QCRH", "QDEL", "QFIN", "QGEN", "QLT", "QNST", "QRVO", "QTWO",
        "QUAD", "QUBT", "QUOT", "R", "RACE", "RAIN", "RAMP", "RARI", "RBA", "RBBN",
        "RBLX", "RC", "RCI", "RCII", "RCL", "RCM", "RCUS", "RDFN", "RDNT", "RDVT",
        "RE", "REAL", "REGI", "REGN", "RERE", "RETA", "REVG", "REX", "REXR", "REZI",
        "RF", "RGA", "RGEN", "RGLD", "RGR", "RH", "RHI", "RHP", "RIG", "RIOT",
        "RIVN", "RJF", "RKLY", "RL", "RLAY", "RLMD", "RLX", "RMBS", "RMD", "RMNI",
        "RNG", "RNST", "ROG", "ROK", "ROKU", "ROL", "ROOT", "ROP", "ROST", "RPD",
        "RPID", "RPM", "RPRX", "RRC", "RRR", "RRX", "RS", "RSG", "RSI", "RSKD"
    ])
    etf_symbols: List[str] = field(default_factory=lambda: [
        "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLU", "XLK",
        "XLP", "XLB", "XLV", "XLI", "XLY", "XLC", "XLRE", "KRE",
        "SMH", "XBI", "GDX",
    ])
    thresholds: List[float] = field(default_factory=lambda: [
        0.000625, 0.00125, 0.0025, 0.005, 0.0075, 0.01, 0.0125,
        0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.10
    ])
    am_thresholds: List[float] = field(default_factory=lambda: [
        0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05
    ])
    update_interval_ms: int = 1000
    cache_refresh_interval: float = 0.5
    history_days: int = 5
    polygon_api_key: str = POLYGON_API_KEY
    polygon_ws_url: str = "wss://delayed.polygon.io/stocks"
    polygon_rest_url: str = "https://api.polygon.io"
    volumes: Dict[str, float] = field(default_factory=dict)
    week52_data: Dict[str, Dict] = field(default_factory=dict)
    fundamental_data: Dict[str, Dict] = field(default_factory=dict)
    fundamental_slopes: Dict[str, Dict] = field(default_factory=dict)
    correlation_data: Dict[str, Dict] = field(default_factory=dict)
    min_tradable_stasis: int = 3
    corr_window_quarters: int = 5


config = Config()
config.symbols = list(dict.fromkeys(config.symbols))

# ============================================================================
# ENUMS & DATA CLASSES
# ============================================================================


class Direction(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class SignalStrength(Enum):
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"
    VERY_STRONG = "VERY_STRONG"


@dataclass
class BitEntry:
    bit: int
    price: float
    timestamp: datetime


@dataclass
class StasisInfo:
    start_time: datetime
    start_price: float
    peak_stasis: int = 1

    def get_duration(self) -> timedelta:
        return datetime.now() - self.start_time

    def get_duration_str(self) -> str:
        t = int(self.get_duration().total_seconds())
        if t < 60:
            return f"{t}s"
        if t < 3600:
            return f"{t // 60}m {t % 60}s"
        return f"{t // 3600}h {(t % 3600) // 60}m"

    def get_start_date_str(self) -> str:
        return self.start_time.strftime("%m/%d %H:%M")

    def get_price_change_pct(self, p: float) -> float:
        return ((p - self.start_price) / self.start_price * 100
                if self.start_price else 0)


# ============================================================================
# HELPERS
# ============================================================================


def calculate_52week_percentile(price, symbol):
    d = config.week52_data.get(symbol)
    if not d:
        return None
    h, l, r = d.get('high'), d.get('low'), d.get('range')
    if not all([h, l, r]) or r <= 0:
        return None
    return max(0, min(100, ((price - l) / r) * 100))


def fmt_slope(v):
    return "—" if v is None else f"{'+' if v >= 0 else ''}{v * 100:.1f}%"


def fmt_rr(rr):
    if rr is None:
        return "—"
    return "0:1" if rr <= 0 else (
        f"{rr:.2f}:1" if rr < 10 else f"{rr:.0f}:1")


def fmt_corr(v):
    if v is None:
        return "—"
    return f"{v:+.3f}"


def fmt_corr_delta(v):
    if v is None:
        return "—"
    arrow = "↘" if v < -0.05 else ("↗" if v > 0.05 else "→")
    return f"{arrow}{v:+.3f}"


# ============================================================================
# PRICE:REVENUE CORRELATION ENGINE
# (Lifted directly from the proven standalone script)
# ============================================================================


def _corr_fetch_financials(symbol: str) -> Optional[Dict]:
    """
    Fetch quarterly financials from Polygon for correlation analysis.
    Returns dict with 'dates' and 'revenue' lists (oldest first).
    Separate from the main fundamental fetch to keep correlation
    self-contained with the exact logic that works in standalone.
    """
    url = (f"{config.polygon_rest_url}/vX/reference/financials"
           f"?ticker={symbol}&timeframe=quarterly&limit=24"
           f"&sort=filing_date&order=desc&apiKey={config.polygon_api_key}")

    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        return None

    results = resp.json().get('results', [])
    if not results:
        return None

    dates = []
    revenue = []

    for r in results:
        try:
            fi = r.get('financials', {})
            inc = fi.get('income_statement', {})
            rev = inc.get('revenues', {}).get('value', 0) or 0
            filing_date = r.get('filing_date', '')

            if filing_date and rev != 0:
                dates.append(filing_date)
                revenue.append(rev)
        except Exception:
            continue

    # Reverse to oldest-first
    dates = dates[::-1]
    revenue = revenue[::-1]

    return {'dates': dates, 'revenue': revenue}


def _corr_fetch_daily_prices(symbol: str, start_date: str,
                             end_date: str) -> pd.DataFrame:
    """
    Fetch daily OHLCV bars from Polygon.
    Returns DataFrame with columns: date, close
    """
    url = (f"{config.polygon_rest_url}/v2/aggs/ticker/{symbol}/range/1/day/"
           f"{start_date}/{end_date}"
           f"?adjusted=true&sort=asc&limit=5000"
           f"&apiKey={config.polygon_api_key}")

    resp = requests.get(url, timeout=20)
    if resp.status_code != 200:
        return pd.DataFrame()

    results = resp.json().get('results', [])
    if not results:
        return pd.DataFrame()

    rows = []
    for bar in results:
        dt = datetime.fromtimestamp(bar['t'] / 1000)
        rows.append({'date': dt.strftime('%Y-%m-%d'), 'close': bar['c']})

    return pd.DataFrame(rows)


def _corr_find_price_on_date(prices_df: pd.DataFrame, target_date: str,
                             max_lookback_days: int = 10) -> Optional[float]:
    """
    Find closing price on or just before target_date.
    Handles weekends/holidays by looking back up to max_lookback_days.
    """
    if prices_df.empty or not target_date:
        return None
    try:
        target = pd.Timestamp(target_date)
    except Exception:
        return None

    df = prices_df.copy()
    df['dt'] = pd.to_datetime(df['date'])

    mask = df['dt'] <= target
    candidates = df[mask]
    if candidates.empty:
        return None

    last_row = candidates.iloc[-1]
    gap = (target - last_row['dt']).days

    if gap <= max_lookback_days:
        return float(last_row['close'])
    return None


def _corr_align_prices_to_quarters(
        financials: Dict, prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each quarterly filing date, find the stock price on that date.
    Returns DataFrame with: quarter, date, revenue, price
    """
    dates = financials['dates']
    revenues = financials['revenue']

    rows = []
    for i in range(len(dates)):
        filing_date = dates[i]
        rev = revenues[i]
        price = _corr_find_price_on_date(prices_df, filing_date)
        rows.append({
            'quarter': i,
            'date': filing_date,
            'revenue': rev,
            'price': price,
        })

    return pd.DataFrame(rows)


def _corr_pearson(prices: np.ndarray,
                  revenues: np.ndarray) -> Optional[float]:
    """Safe Pearson correlation."""
    if len(prices) != len(revenues) or len(prices) < 3:
        return None
    if np.std(prices) < 1e-10 or np.std(revenues) < 1e-10:
        return None
    try:
        c = np.corrcoef(prices, revenues)[0, 1]
        return None if (np.isnan(c) or np.isinf(c)) else round(float(c), 4)
    except Exception:
        return None


def _corr_fetch_current_price(symbol: str) -> Optional[float]:
    """Get the most recent closing price."""
    end = datetime.now()
    start = end - timedelta(days=7)
    url = (f"{config.polygon_rest_url}/v2/aggs/ticker/{symbol}/range/1/day/"
           f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
           f"?adjusted=true&sort=desc&limit=5"
           f"&apiKey={config.polygon_api_key}")
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            results = resp.json().get('results', [])
            if results:
                return results[0]['c']
    except Exception:
        pass
    return None


def calculate_symbol_correlation(symbol: str) -> Optional[Dict]:
    """
    Full correlation pipeline for one symbol.
    This is the EXACT logic from the standalone script that works.
    Returns the result dict or None.
    """
    window = config.corr_window_quarters

    # 1. Fetch financials
    financials = _corr_fetch_financials(symbol)
    if not financials or len(financials['revenue']) < window:
        return None
    time.sleep(0.15)

    # 2. Determine date range for price data
    earliest_date = financials['dates'][0]
    try:
        earliest_dt = (datetime.strptime(earliest_date, '%Y-%m-%d')
                       - timedelta(days=30))
        start_date = earliest_dt.strftime('%Y-%m-%d')
    except Exception:
        start_date = '2020-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')

    # 3. Fetch daily prices
    prices_df = _corr_fetch_daily_prices(symbol, start_date, end_date)
    if prices_df.empty:
        return None
    time.sleep(0.15)

    # 4. Align prices to filing dates
    aligned = _corr_align_prices_to_quarters(financials, prices_df)

    # 5. Get current price
    current_price = _corr_fetch_current_price(symbol)
    if current_price is None and not prices_df.empty:
        current_price = prices_df.iloc[-1]['close']
    if current_price is None:
        return None
    time.sleep(0.15)

    # 6. Calculate correlations
    result = {
        'symbol': symbol,
        'corr_at_earnings': None,
        'corr_now': None,
        'corr_delta': None,
        'decorrelation_score': None,
        'price_vs_rev_divergence': None,
        'earnings_date_price': None,
        'current_price': current_price,
        'price_change_since_earnings_pct': None,
        'latest_rev_change_pct': None,
        'aligned_count': 0,
    }

    # Drop rows with missing prices
    valid = aligned.dropna(subset=['price']).copy().reset_index(drop=True)
    n = len(valid)
    result['aligned_count'] = n

    if n < window:
        return result

    # Use the last `window` quarters
    tail = valid.tail(window).copy()
    trail_prices = tail['price'].values.astype(float)
    trail_revs = tail['revenue'].values.astype(float)

    earnings_price = float(trail_prices[-1])
    result['earnings_date_price'] = round(earnings_price, 2)

    # -- CORRELATION AT EARNINGS (all original prices) --
    corr_earn = _corr_pearson(trail_prices, trail_revs)
    result['corr_at_earnings'] = corr_earn

    if corr_earn is None:
        return result

    # -- CORRELATION NOW (replace last price with current live price) --
    live_prices = trail_prices.copy()
    live_prices[-1] = current_price

    corr_now = _corr_pearson(live_prices, trail_revs)
    result['corr_now'] = corr_now

    # -- DELTA --
    if corr_earn is not None and corr_now is not None:
        result['corr_delta'] = round(corr_now - corr_earn, 4)

    # -- PRICE CHANGE SINCE EARNINGS --
    if earnings_price > 0:
        result['price_change_since_earnings_pct'] = round(
            ((current_price - earnings_price) / earnings_price) * 100, 2)

    # -- LATEST REVENUE CHANGE --
    if n >= 2:
        rev_prev = valid.iloc[-2]['revenue']
        rev_curr = valid.iloc[-1]['revenue']
        if rev_prev and rev_prev != 0:
            result['latest_rev_change_pct'] = round(
                ((rev_curr - rev_prev) / abs(rev_prev)) * 100, 2)

    # -- DECORRELATION SCORE (0 to 1) --
    if corr_now is not None:
        abs_component = max(0, (1.0 - abs(corr_now)) * 0.5)
        delta_component = 0.0
        if result['corr_delta'] is not None:
            delta_component = max(0, min(0.5, (-result['corr_delta']) * 0.5))
        result['decorrelation_score'] = round(
            abs_component + delta_component, 4)

    # -- DIVERGENCE DIRECTION --
    pchg = result.get('price_change_since_earnings_pct')
    rchg = result.get('latest_rev_change_pct')

    if pchg is not None and rchg is not None:
        if pchg > rchg + 10:
            result['price_vs_rev_divergence'] = 'PRICE_AHEAD'
        elif rchg > pchg + 10:
            result['price_vs_rev_divergence'] = 'PRICE_BEHIND'
        else:
            result['price_vs_rev_divergence'] = 'ALIGNED'
    elif pchg is not None:
        if abs(pchg) > 15:
            result['price_vs_rev_divergence'] = (
                'PRICE_AHEAD' if pchg > 0 else 'PRICE_BEHIND')
        else:
            result['price_vs_rev_divergence'] = 'ALIGNED'

    return result


def calculate_all_correlations():
    """
    Calculate price:revenue correlations for all non-ETF symbols.
    Uses the exact same pipeline as the standalone script.
    """
    print("\n📊 CALCULATING PRICE:REVENUE CORRELATIONS...")
    print(f"   Window: {config.corr_window_quarters} quarters")
    ok = fail = skip = 0

    for i, sym in enumerate(config.symbols):
        # Skip ETFs
        if sym in config.etf_symbols:
            skip += 1
            continue

        try:
            result = calculate_symbol_correlation(sym)

            if result and result.get('corr_at_earnings') is not None:
                config.correlation_data[sym] = result
                ok += 1

                delta = result.get('corr_delta')
                decor = result.get('decorrelation_score', 0)
                ep = result.get('earnings_date_price', '?')
                cp = result.get('current_price', '?')
                pchg = result.get('price_change_since_earnings_pct')
                div = result.get('price_vs_rev_divergence', '?')
                aligned = result.get('aligned_count', 0)

                flag = "🔔" if decor and decor > 0.3 else "  "
                delta_s = fmt_corr_delta(delta)
                pchg_s = f"{pchg:+.1f}%" if pchg is not None else "?"

                print(f"   {flag} {sym:6s}: "
                      f"@earn={result['corr_at_earnings']:+.3f} "
                      f"@now={result.get('corr_now', 0):+.3f}"
                      f"  Δ={delta_s:>8s}"
                      f"  (${ep}→${cp} {pchg_s})"
                      f"  decor={decor:.3f}"
                      f"  [{div:>12s}]"
                      f"  {aligned}q")
            elif result:
                config.correlation_data[sym] = result
                fail += 1
                print(f"      ✗ {sym:6s}: "
                      f"aligned={result.get('aligned_count', 0)}q "
                      f"(need {config.corr_window_quarters})")
            else:
                fail += 1
                print(f"      ✗ {sym:6s}: no data")

        except Exception as e:
            print(f"      ⚠️ {sym} error: {e}")
            fail += 1

        if (i + 1) % 10 == 0:
            print(f"   --- Progress: {i + 1}/{len(config.symbols)} "
                  f"(✓{ok} ✗{fail} ⏭{skip})")

    print(f"\n✅ Correlations: {ok} calculated, {fail} failed, "
          f"{skip} skipped (ETFs)\n")


# ============================================================================
# FUNDAMENTAL DATA
# ============================================================================


def fetch_fundamental_data_polygon(sym):
    try:
        url = (f"{config.polygon_rest_url}/vX/reference/financials"
               f"?ticker={sym}&timeframe=quarterly&limit=24"
               f"&sort=filing_date&order=desc"
               f"&apiKey={config.polygon_api_key}")
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            return None
        results = resp.json().get('results', [])
        if not results:
            return None
        fund = {k: [] for k in [
            'dates', 'revenue', 'net_income', 'operating_cash_flow',
            'capex', 'fcf', 'total_assets', 'total_liabilities',
            'shareholders_equity', 'current_assets', 'current_liabilities',
            'total_debt', 'eps']}
        for r in results:
            try:
                fi = r.get('financials', {})
                inc = fi.get('income_statement', {})
                cf = fi.get('cash_flow_statement', {})
                bs = fi.get('balance_sheet', {})
                rev = inc.get('revenues', {}).get('value', 0) or 0
                ni = inc.get('net_income_loss', {}).get('value', 0) or 0
                eps = (inc.get('basic_earnings_per_share', {})
                       .get('value', 0) or 0)
                ocf = (cf.get('net_cash_flow_from_operating_activities', {})
                       .get('value', 0) or 0)
                cx = (cf.get('net_cash_flow_from_investing_activities', {})
                      .get('value', 0) or 0)
                ta = bs.get('assets', {}).get('value', 0) or 0
                tl = bs.get('liabilities', {}).get('value', 0) or 0
                eq = bs.get('equity', {}).get('value', 0) or 0
                ca = bs.get('current_assets', {}).get('value', 0) or 0
                cl = bs.get('current_liabilities', {}).get('value', 0) or 0
                ltd = bs.get('long_term_debt', {}).get('value', 0) or 0
                std = bs.get('short_term_debt', {}).get('value', 0) or 0
                fund['dates'].append(r.get('filing_date', ''))
                fund['revenue'].append(rev)
                fund['net_income'].append(ni)
                fund['operating_cash_flow'].append(ocf)
                fund['capex'].append(abs(cx))
                fund['fcf'].append(ocf + cx)
                fund['total_assets'].append(ta)
                fund['total_liabilities'].append(tl)
                fund['shareholders_equity'].append(eq)
                fund['current_assets'].append(ca)
                fund['current_liabilities'].append(cl)
                fund['total_debt'].append(ltd + std)
                fund['eps'].append(eps)
            except Exception:
                continue
        for k in fund:
            fund[k] = fund[k][::-1]
        return fund
    except Exception:
        return None


def calculate_slopes(series, ss=4, sl=20):
    if not series or len(series) < 5:
        return None, None
    s = pd.Series(series).replace([np.inf, -np.inf], np.nan)
    s5 = s20 = None
    try:
        if len(s.dropna()) >= 5:
            e = s.ewm(span=ss, adjust=False).mean()
            if abs(e.iloc[-5]) > 0.0001:
                s5 = (e.iloc[-1] - e.iloc[-5]) / abs(e.iloc[-5])
    except Exception:
        pass
    try:
        if len(s.dropna()) >= 21:
            e = s.ewm(span=sl, adjust=False).mean()
            if abs(e.iloc[-21]) > 0.0001:
                s20 = (e.iloc[-1] - e.iloc[-21]) / abs(e.iloc[-21])
    except Exception:
        pass
    return s5, s20


def calculate_all_slopes(fund, ratios):
    sl = {}
    sl['Rev_Slope_5'], sl['Rev_Slope_20'] = calculate_slopes(
        fund.get('revenue', []))
    sl['FCF_Slope_5'], sl['FCF_Slope_20'] = calculate_slopes(
        fund.get('fcf', []))
    for n, k in [('P/E Ratio', 'pe_ratio'), ('Return on Equity', 'roe'),
                 ('Net Profit Margin', 'net_profit_margin'),
                 ('Debt to Equity Ratio', 'debt_to_equity')]:
        sl[f'{n}_Slope_5'], sl[f'{n}_Slope_20'] = calculate_slopes(
            ratios.get(k, []))
    fl = ratios.get('fcfy', [])
    sl['FCFY'] = fl[-1] if fl and fl[-1] is not None else None
    return sl


def fetch_all_fundamental_data():
    print("\n📊 FETCHING FUNDAMENTAL DATA...")
    ok = fail = 0
    for i, sym in enumerate(config.symbols):
        try:
            fund = fetch_fundamental_data_polygon(sym)
            if fund and len(fund.get('revenue', [])) >= 4:
                price = 100
                w = config.week52_data.get(sym, {})
                if w.get('high') and w.get('low'):
                    price = (w['high'] + w['low']) / 2
                eq = fund['shareholders_equity'][-1]
                mcap = eq * 2 if eq and eq > 0 else 1e9
                ratios = {k: [] for k in [
                    'pe_ratio', 'roe', 'net_profit_margin',
                    'debt_to_equity', 'fcfy']}
                for j in range(len(fund['revenue'])):
                    try:
                        eps = fund['eps'][j]
                        ratios['pe_ratio'].append(
                            price / eps if eps > 0 else None)
                        eq_j = fund['shareholders_equity'][j]
                        ratios['roe'].append(
                            fund['net_income'][j] / eq_j
                            if eq_j > 0 else None)
                        rev_j = fund['revenue'][j]
                        ratios['net_profit_margin'].append(
                            fund['net_income'][j] / rev_j
                            if rev_j else None)
                        ratios['debt_to_equity'].append(
                            fund['total_debt'][j] / eq_j
                            if eq_j > 0 else None)
                        if j >= 3:
                            ratios['fcfy'].append(
                                sum(fund['fcf'][max(0, j - 3):j + 1]) / mcap
                                if mcap else None)
                        else:
                            ratios['fcfy'].append(None)
                    except Exception:
                        for k in ratios:
                            ratios[k].append(None)
                slopes = calculate_all_slopes(fund, ratios)
                config.fundamental_data[sym] = fund
                config.fundamental_slopes[sym] = slopes
                ok += 1
            else:
                fail += 1
        except Exception:
            fail += 1
        if (i + 1) % 25 == 0:
            print(f"   📈 {i + 1}/{len(config.symbols)} (✓{ok} ✗{fail})")
        time.sleep(0.15)
    print(f"✅ Fundamentals: {ok} ok, {fail} failed\n")


# ============================================================================
# MERIT SCORING
# ============================================================================


def calculate_stasis_merit_score(snap):
    ms = 0
    st = snap.get('stasis', 0)
    for t, p in [(15, 10), (12, 9), (10, 8), (8, 7), (7, 6),
                 (6, 5), (5, 4), (4, 3), (3, 2), (2, 1)]:
        if st >= t:
            ms += p
            break
    rr = snap.get('risk_reward')
    if rr:
        for t, p in [(3, 5), (2.5, 4), (2, 3), (1.5, 2), (1, 1)]:
            if rr >= t:
                ms += p
                break
    ms += {'VERY_STRONG': 4, 'STRONG': 3, 'MODERATE': 2, 'WEAK': 1}.get(
        snap.get('signal_strength', ''), 0)
    dur = snap.get('duration_seconds', 0)
    if dur >= 3600:
        ms += 3
    elif dur >= 1800:
        ms += 2
    elif dur >= 900:
        ms += 1
    return ms


def calculate_correlation_merit_score(
        symbol: str, direction: Optional[str]) -> Tuple[int, Dict]:
    """
    Merit score from price:revenue decorrelation.

    Uses the EXACT keys produced by calculate_symbol_correlation():
      corr_at_earnings, corr_now, corr_delta, decorrelation_score,
      price_vs_rev_divergence

    Scoring:
      Decorrelation score tiers:    0-6 pts
      Correlation delta rate:       0-4 pts
      Direction alignment bonus:    -1 to +3 pts
    """
    score = 0
    details = {
        'corr_at_earnings': None,
        'corr_now': None,
        'corr_delta': None,
        'decor_score': None,
        'divergence': None,
        'corr_merit': 0,
    }

    corr = config.correlation_data.get(symbol)
    if not corr or corr.get('corr_at_earnings') is None:
        return score, details

    details['corr_at_earnings'] = corr['corr_at_earnings']
    details['corr_now'] = corr.get('corr_now')
    details['corr_delta'] = corr.get('corr_delta')
    details['decor_score'] = corr.get('decorrelation_score')
    details['divergence'] = corr.get('price_vs_rev_divergence')

    # --- Component 1: Decorrelation Score (0-1) → 0-6 points ---
    decor = corr.get('decorrelation_score', 0) or 0
    for threshold, points in [
        (0.7, 6), (0.55, 5), (0.4, 4), (0.3, 3), (0.2, 2), (0.1, 1)
    ]:
        if decor >= threshold:
            score += points
            break

    # --- Component 2: Correlation Delta → 0-4 points ---
    delta = corr.get('corr_delta')
    if delta is not None:
        for threshold, points in [
            (-0.4, 4), (-0.25, 3), (-0.15, 2), (-0.05, 1)
        ]:
            if delta <= threshold:
                score += points
                break

    # --- Component 3: Direction Alignment → -1 to +3 points ---
    divergence = corr.get('price_vs_rev_divergence')
    if divergence and direction:
        if divergence == 'PRICE_BEHIND' and direction == 'LONG':
            score += 3
        elif divergence == 'PRICE_AHEAD' and direction == 'SHORT':
            score += 3
        elif divergence == 'PRICE_AHEAD' and direction == 'LONG':
            score -= 1
        elif divergence == 'PRICE_BEHIND' and direction == 'SHORT':
            score -= 1

    score = max(0, score)
    details['corr_merit'] = score
    return score, details


def calculate_fundamental_merit_score(symbol, w52_pct):
    ms = 0
    sd = {}
    slopes = config.fundamental_slopes.get(symbol, {})
    fund = config.fundamental_data.get(symbol, {})

    # ---------- helper ----------
    def last5_all_positive(series):
        if len(series) < 5:
            return False
        last5 = series[-5:]
        return all(x is not None and x > 0 for x in last5)
    
    def last5_latest_is_highest(series):
        if len(series) < 5:
            return False
        last5 = series[-5:]
        if any(x is None for x in last5):
            return False
        return last5[-1] == max(last5)
    
    # ---------- NET INCOME ----------
    ni = fund.get('net_income', [])
    if last5_all_positive(ni):
        ms += 4
        sd['NI_5_Positive'] = True
    else:
        sd['NI_5_Positive'] = False
    
    if last5_latest_is_highest(ni):
        ms += 3
        sd['NI_Latest_Highest_5Q'] = True
    else:
        sd['NI_Latest_Highest_5Q'] = False
    
    # ---------- OPERATING CASH FLOW ----------
    ocf = fund.get('operating_cash_flow', [])
    if last5_all_positive(ocf):
        ms += 4
        sd['OCF_5_Positive'] = True
    else:
        sd['OCF_5_Positive'] = False
    
    if last5_latest_is_highest(ocf):
        ms += 3
        sd['OCF_Latest_Highest_5Q'] = True
    else:
        sd['OCF_Latest_Highest_5Q'] = False
    
    # ---------- FREE CASH FLOW ----------
    fcf = fund.get('fcf', [])
    if last5_all_positive(fcf):
        ms += 4
        sd['FCF_5_Positive'] = True
    else:
        sd['FCF_5_Positive'] = False
    
    if last5_latest_is_highest(fcf):
        ms += 3
        sd['FCF_Latest_Highest_5Q'] = True
    else:
        sd['FCF_Latest_Highest_5Q'] = False
    
    # ---------- REVENUE ----------
    rev = fund.get('revenue', [])
    if last5_all_positive(rev):
        ms += 2
        sd['REV_5_Positive'] = True
    else:
        sd['REV_5_Positive'] = False
    
    if last5_latest_is_highest(rev):
        ms += 3
        sd['REV_Latest_Highest_5Q'] = True
    else:
        sd['REV_Latest_Highest_5Q'] = False
    
    if not slopes:
        if w52_pct is not None:
            for t, p in [(5, 8), (15, 7), (25, 6), (35, 5), (45, 4),
                         (55, 3), (65, 2), (75, 1)]:
                if w52_pct <= t:
                    ms += p
                    break
        return ms, sd
    for lbl, key, tps in [
        ('Rev_5', 'Rev_Slope_5',
         [(0.30, 4), (0.20, 3), (0.10, 2), (0.05, 1)]),
        ('FCF_5', 'FCF_Slope_5',
         [(0.40, 4), (0.25, 3), (0.10, 2), (0.05, 1)]),
        ('ROE_5', 'Return on Equity_Slope_5', [(0.20, 2), (0.10, 1)]),
        ('NPM_5', 'Net Profit Margin_Slope_5', [(0.20, 2), (0.10, 1)])]:
        v = slopes.get(key)
        sd[lbl] = v
        if v is not None:
            for t, p in tps:
                if v >= t:
                    ms += p
                    break
    for lbl, key, tps in [
        ('PE_5', 'P/E Ratio_Slope_5',
         [(-0.25, 3), (-0.15, 2), (-0.05, 1)]),
        ('DE_5', 'Debt to Equity Ratio_Slope_5',
         [(-0.20, 2), (-0.10, 1)])]:
        v = slopes.get(key)
        sd[lbl] = v
        if v is not None:
            for t, p in tps:
                if v <= t:
                    ms += p
                    break
    if w52_pct is not None:
        for t, p in [(5, 8), (15, 7), (25, 6), (35, 5), (45, 4),
                     (55, 3), (65, 2), (75, 1)]:
            if w52_pct <= t:
                ms += p
                break
    fcfy = slopes.get('FCFY')
    sd['FCFY'] = fcfy
    if fcfy is not None:
        if fcfy >= 0.15:
            ms += 3
        elif fcfy >= 0.10:
            ms += 2
        elif fcfy >= 0.05:
            ms += 1
    
    return ms, sd


# ============================================================================
# DATA FETCHERS
# ============================================================================


def fetch_52_week_data():
    print("📊 Fetching 52-week data...")
    w52 = {}
    end = datetime.now()
    start = end - timedelta(days=365)
    ok = fail = 0
    for i, sym in enumerate(config.symbols):
        try:
            url = (
                f"{config.polygon_rest_url}/v2/aggs/ticker/{sym}/range/1/day/"
                f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
                f"?adjusted=true&sort=asc&limit=365"
                f"&apiKey={config.polygon_api_key}")
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                res = r.json().get('results', [])
                if res:
                    hv = max(b['h'] for b in res)
                    lv = min(b['l'] for b in res)
                    w52[sym] = {'high': hv, 'low': lv,
                                'range': hv - lv, 'current': res[-1]['c']}
                    ok += 1
                else:
                    w52[sym] = {'high': None, 'low': None,
                                'range': None, 'current': None}
                    fail += 1
            else:
                w52[sym] = {'high': None, 'low': None,
                            'range': None, 'current': None}
                fail += 1
            if (i + 1) % 50 == 0:
                print(f"   52W: {i + 1}/{len(config.symbols)} "
                      f"(✓{ok} ✗{fail})")
            time.sleep(0.12)
        except Exception:
            w52[sym] = {'high': None, 'low': None,
                        'range': None, 'current': None}
            fail += 1
    print(f"✅ 52-week: {ok} ok, {fail} failed\n")
    return w52


def fetch_volume_data():
    print("📊 Fetching volume data...")
    vols = {}
    end = datetime.now()
    start = end - timedelta(days=45)
    for i, sym in enumerate(config.symbols):
        try:
            url = (
                f"{config.polygon_rest_url}/v2/aggs/ticker/{sym}/range/1/day/"
                f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
                f"?adjusted=true&sort=desc&limit=30"
                f"&apiKey={config.polygon_api_key}")
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                res = r.json().get('results', [])
                if res:
                    vols[sym] = (
                        sum(b['v'] for b in res) / len(res)) / 1e6
                else:
                    vols[sym] = 10.0
            else:
                vols[sym] = 10.0
            if (i + 1) % 50 == 0:
                print(f"   Vol: {i + 1}/{len(config.symbols)}")
            time.sleep(0.12)
        except Exception:
            vols[sym] = 10.0
    print("✅ Volume loaded\n")
    return vols


def fetch_historical_bars(sym, days=5):
    bars = []
    end = datetime.now()
    start = end - timedelta(days=days)
    try:
        url = (
            f"{config.polygon_rest_url}/v2/aggs/ticker/{sym}/range/1/minute/"
            f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
            f"?adjusted=true&sort=asc&limit=50000"
            f"&apiKey={config.polygon_api_key}")
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            res = r.json().get('results', [])
            bars = [{'timestamp': datetime.fromtimestamp(b['t'] / 1000),
                     'close': b['c']} for b in res]
    except Exception:
        pass
    return bars


# ============================================================================
# BITSTREAM
# ============================================================================


class Bitstream:
    def __init__(self, symbol, threshold, initial_price, volume):
        self.symbol = symbol
        self.threshold = threshold
        self.initial_price = initial_price
        self.volume = volume
        self.is_etf = symbol in config.etf_symbols
        self.reference_price = initial_price
        self.current_live_price = initial_price
        self.last_price_update = datetime.now()
        self._update_bands()
        self.bits: deque = deque(maxlen=500)
        self.current_stasis = 0
        self.last_bit = None
        self.direction = None
        self.signal_strength = None
        self.stasis_info: Optional[StasisInfo] = None
        self.total_bits = 0
        self._lock = threading.Lock()

    def _update_bands(self):
        self.band_width = self.threshold * self.reference_price
        self.upper_band = self.reference_price + self.band_width
        self.lower_band = self.reference_price - self.band_width

    def process_price(self, price, timestamp):
        with self._lock:
            self.current_live_price = price
            self.last_price_update = timestamp
            if self.lower_band < price < self.upper_band:
                return
            if self.band_width <= 0:
                return
            x = int((price - self.reference_price) / self.band_width)
            if x > 0:
                for _ in range(x):
                    self.bits.append(BitEntry(1, price, timestamp))
                    self.total_bits += 1
                self.reference_price = price
                self._update_bands()
            elif x < 0:
                for _ in range(abs(x)):
                    self.bits.append(BitEntry(0, price, timestamp))
                    self.total_bits += 1
                self.reference_price = price
                self._update_bands()
            self._update_stasis(timestamp)

    def _update_stasis(self, ts):
        if len(self.bits) < 2:
            self.current_stasis = len(self.bits)
            self.last_bit = self.bits[-1].bit if self.bits else None
            self.direction = None
            self.signal_strength = None
            return
        bl = list(self.bits)
        sc = 1
        si = len(bl) - 1
        for i in range(len(bl) - 1, 0, -1):
            if bl[i].bit != bl[i - 1].bit:
                sc += 1
                si = i - 1
            else:
                break
        prev = self.current_stasis
        self.current_stasis = sc
        self.last_bit = bl[-1].bit
        if prev < 2 and sc >= 2 and 0 <= si < len(bl):
            self.stasis_info = StasisInfo(
                bl[si].timestamp, bl[si].price, sc)
        elif (sc >= 2 and self.stasis_info
              and sc > self.stasis_info.peak_stasis):
            self.stasis_info.peak_stasis = sc
        elif prev >= 2 and sc < 2:
            self.stasis_info = None
        if sc >= 2:
            self.direction = (Direction.LONG if self.last_bit == 0
                              else Direction.SHORT)
            if sc >= 10:
                self.signal_strength = SignalStrength.VERY_STRONG
            elif sc >= 7:
                self.signal_strength = SignalStrength.STRONG
            elif sc >= 5:
                self.signal_strength = SignalStrength.MODERATE
            elif sc >= 3:
                self.signal_strength = SignalStrength.WEAK
            else:
                self.signal_strength = None
        else:
            self.direction = None
            self.signal_strength = None

    def get_snapshot(self, live_price=None):
        with self._lock:
            p = (live_price if live_price is not None
                 else self.current_live_price)
            si = self.stasis_info
            tp = sl = rr = None
            distance_to_tp_pct = distance_to_sl_pct = None
            stasis_price_change_pct = None
            if si is not None:
                stasis_price_change_pct = si.get_price_change_pct(p)
            if self.direction and self.current_stasis >= 2:
                if self.direction == Direction.LONG:
                    tp, sl = self.upper_band, self.lower_band
                    reward, risk = tp - p, p - sl
                else:
                    tp, sl = self.lower_band, self.upper_band
                    reward, risk = p - tp, sl - p
                if risk > 0 and reward > 0:
                    rr = reward / risk
                elif risk > 0:
                    rr = 0.0
                if p > 0:
                    distance_to_tp_pct = (abs(tp - p) / p) * 100
                    distance_to_sl_pct = (abs(sl - p) / p) * 100
            return {
                'symbol': self.symbol, 'is_etf': self.is_etf,
                'threshold': self.threshold,
                'threshold_pct': self.threshold * 100,
                'stasis': self.current_stasis,
                'total_bits': self.total_bits,
                'current_price': p,
                'anchor_price': si.start_price if si else None,
                'direction': (self.direction.value
                              if self.direction else None),
                'signal_strength': (self.signal_strength.value
                                    if self.signal_strength else None),
                'is_tradable': (
                    self.current_stasis >= config.min_tradable_stasis
                    and self.direction is not None
                    and self.volume > 1.0),
                'stasis_start_str': (si.get_start_date_str()
                                     if si else "—"),
                'stasis_duration_str': (si.get_duration_str()
                                        if si else "—"),
                'duration_seconds': (si.get_duration().total_seconds()
                                     if si else 0),
                'stasis_price_change_pct': stasis_price_change_pct,
                'take_profit': tp, 'stop_loss': sl, 'risk_reward': rr,
                'distance_to_tp_pct': distance_to_tp_pct,
                'distance_to_sl_pct': distance_to_sl_pct,
                'week52_percentile': calculate_52week_percentile(
                    p, self.symbol),
                'volume': self.volume,
            }


# ============================================================================
# PRICE FEED
# ============================================================================


class PolygonPriceFeed:
    def __init__(self):
        self.lock = threading.Lock()
        self.current_prices = {s: None for s in config.symbols}
        self.is_running = False
        self.ws = None
        self.message_count = 0

    def start(self):
        self.is_running = True
        threading.Thread(target=self._loop, daemon=True).start()
        print("✅ WebSocket starting...")

    def _loop(self):
        while self.is_running:
            try:
                self._connect()
            except Exception as e:
                print(f"WS err: {e}")
                time.sleep(5)

    def _connect(self):
        def on_msg(ws, msg):
            try:
                data = json.loads(msg)
                for m in (data if isinstance(data, list) else [data]):
                    self._proc(m)
            except Exception:
                pass

        def on_open(ws):
            print("✅ WS connected")
            ws.send(json.dumps({
                "action": "auth",
                "params": config.polygon_api_key}))

        self.ws = websocket.WebSocketApp(
            config.polygon_ws_url,
            on_open=on_open, on_message=on_msg)
        self.ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

    def _proc(self, msg):
        if (msg.get('ev') == 'status'
                and msg.get('status') == 'auth_success'):
            self._sub()
        elif msg.get('ev') in ('A', 'AM', 'T', 'Q'):
            sym = msg.get('sym', '') or msg.get('S', '')
            price = (msg.get('c') or msg.get('vw')
                     or msg.get('p') or msg.get('bp'))
            if price and sym in self.current_prices:
                with self.lock:
                    self.current_prices[sym] = float(price)
                    self.message_count += 1

    def _sub(self):
        for i in range(0, len(config.symbols), 50):
            batch = config.symbols[i:i + 50]
            self.ws.send(json.dumps({
                "action": "subscribe",
                "params": ",".join(f"A.{s}" for s in batch)}))
            time.sleep(0.1)
        print(f"📡 Subscribed {len(config.symbols)} symbols")

    def get_prices(self):
        with self.lock:
            return {k: v for k, v in self.current_prices.items() if v}

    def get_status(self):
        with self.lock:
            return {
                'connected': sum(
                    1 for v in self.current_prices.values() if v),
                'total': len(config.symbols),
                'messages': self.message_count}


price_feed = PolygonPriceFeed()


# ============================================================================
# BITSTREAM MANAGER
# ============================================================================


class BitstreamManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.streams: Dict[Tuple[str, float], Bitstream] = {}
        self.is_running = False
        self.cached_am_data: List[Dict] = []
        self.cache_lock = threading.Lock()
        self.initialized = False
        self.backfill_complete = False
        self.backfill_progress = 0

    def backfill(self):
        print("\n" + "=" * 60 + "\n📜 BACKFILLING\n" + "=" * 60)
        hist = {}
        for i, sym in enumerate(config.symbols):
            bars = fetch_historical_bars(sym, config.history_days)
            if bars:
                hist[sym] = bars
            self.backfill_progress = int(
                (i + 1) / len(config.symbols) * 100)
            if (i + 1) % 25 == 0:
                print(f"   📊 {i + 1}/{len(config.symbols)} "
                      f"({self.backfill_progress}%)")
            time.sleep(0.12)
        with self.lock:
            for sym, bars in hist.items():
                if not bars:
                    continue
                vol = config.volumes.get(sym, 10.0)
                for th in config.thresholds:
                    key = (sym, th)
                    self.streams[key] = Bitstream(
                        sym, th, bars[0]['close'], vol)
                    for bar in bars:
                        self.streams[key].process_price(
                            bar['close'], bar['timestamp'])
        self.initialized = True
        self.backfill_complete = True
        tradable = sum(
            1 for s in self.streams.values()
            if s.current_stasis >= config.min_tradable_stasis
            and s.direction is not None and s.volume > 1.0)
        print(f"✅ Streams: {len(self.streams)} | Tradable: {tradable}")
        print("=" * 60)

    def start(self):
        self.is_running = True
        threading.Thread(target=self._process, daemon=True).start()
        threading.Thread(target=self._cache, daemon=True).start()

    def _process(self):
        while self.is_running:
            time.sleep(0.1)
            if not self.backfill_complete:
                continue
            prices = price_feed.get_prices()
            ts = datetime.now()
            with self.lock:
                for sym, p in prices.items():
                    for th in config.thresholds:
                        k = (sym, th)
                        if k in self.streams:
                            self.streams[k].process_price(p, ts)

    def _cache(self):
        while self.is_running:
            time.sleep(config.cache_refresh_interval)
            if not self.initialized:
                continue
            prices = price_feed.get_prices()
            snaps = []
            with self.lock:
                for s in self.streams.values():
                    snaps.append(
                        s.get_snapshot(prices.get(s.symbol)))
            am = self._build_am(snaps)
            with self.cache_lock:
                self.cached_am_data = am

    def _build_am(self, snaps):
        rows = []
        for s in snaps:
            if s['threshold'] not in config.am_thresholds:
                continue

            sms = calculate_stasis_merit_score(s)
            fms, sd = calculate_fundamental_merit_score(
                s['symbol'], s.get('week52_percentile'))

            # Correlation merit score
            cms, cd = calculate_correlation_merit_score(
                s['symbol'], s.get('direction'))

            tms = sms + fms + cms

            rows.append({
                **s,
                'sms': sms,
                'fms': fms,
                'cms': cms,
                'tms': tms,
                'slope_details': sd,
                'corr_details': cd,
            })
        return rows

    def get_am_data(self):
        with self.cache_lock:
            return copy.deepcopy(self.cached_am_data)


manager = BitstreamManager()

# ============================================================================
# DASH APP
# ============================================================================

AM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;600&display=swap');
body { background: #f5f0e8 !important; }
.title-font { font-family: 'Orbitron', sans-serif !important; }
"""

app = dash.Dash(
    __name__, external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True)
app.title = "STASIS AM"
server = app.server

app.index_string = '''<!DOCTYPE html>
<html><head>
{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>''' + AM_CSS + '''</style>
</head><body>
{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body></html>'''

app.layout = html.Div([
    dcc.Store(id='fmode', data='tradable'),
    dcc.Interval(id='tick', interval=1000, n_intervals=0),

    # Header
    html.Div([
        html.Span("📈", style={'fontSize': '22px'}),
        html.Span(" STASIS AM", className="title-font ms-2",
                  style={'fontSize': '16px', 'fontWeight': '700',
                         'color': '#1a5c2a', 'letterSpacing': '2px'}),
        html.Span(" — ALPHA MARKETS", className="title-font",
                  style={'fontSize': '9px', 'color': '#888',
                         'letterSpacing': '1px'}),
    ], style={'padding': '8px'}),

    html.Div(id='status',
             style={'fontSize': '10px', 'padding': '4px 8px',
                    'background': '#e8f5e9', 'fontWeight': 'bold'}),

    # Filters
    html.Div([
        dbc.ButtonGroup([
            dbc.Button("ALL", id="f-all", size="sm", outline=True,
                       style={'fontSize': '9px'}),
            dbc.Button("TRADABLE", id="f-trad", size="sm",
                       outline=True, active=True,
                       style={'fontSize': '9px', 'color': '#1a5c2a'}),
            dbc.Button("DECORR", id="f-decorr", size="sm",
                       outline=True,
                       style={'fontSize': '9px', 'color': '#8b4513'},
                       className="ms-1"),
        ], size="sm", className="me-2"),
        dcc.Dropdown(
            id='f-dir',
            options=[{'label': x, 'value': x}
                     for x in ['ALL', 'LONG', 'SHORT']],
            value='ALL', clearable=False,
            style={'width': '80px', 'fontSize': '10px',
                   'display': 'inline-block'}),
        dcc.Dropdown(id='f-sort', options=[
            {'label': 'TMS ↓', 'value': 'tms'},
            {'label': 'FMS ↓', 'value': 'fms'},
            {'label': 'CMS ↓', 'value': 'cms'},
            {'label': 'DECORR ↓', 'value': 'decorr'},
            {'label': 'Δ CORR ↑', 'value': 'corr_delta'},
            {'label': 'STASIS ↓', 'value': 'stasis'},
            {'label': '52W ↑', 'value': '52w'},
        ], value='tms', clearable=False,
            style={'width': '100px', 'fontSize': '10px',
                   'display': 'inline-block', 'marginLeft': '4px'}),
    ], className="d-flex align-items-center p-1",
       style={'background': '#f5f0e8'}),

    # Table
    dash_table.DataTable(
        id='tbl',
        columns=[{'name': c, 'id': c} for c in [
            '✓', 'SYM', 'BAND', 'STS', 'DIR',
            'SMS', 'FMS', 'CMS', 'TMS',
            'C@E', 'C@N', 'ΔCOR', 'DCOR', 'DIV',
            'REV5', 'FCF5', 'FCFY', '52W',
            'PRICE', 'TP', 'SL', 'R:R', 'DUR']],
        sort_action='native',
        style_table={'overflowY': 'auto', 'height': '80vh'},
        style_cell={
            'backgroundColor': '#faf7f0', 'color': '#1a1a1a',
            'padding': '3px 4px', 'fontSize': '10px',
            'fontFamily': 'Consolas, monospace',
            'whiteSpace': 'nowrap', 'textAlign': 'right',
            'border': '1px solid #ddd'},
        style_cell_conditional=[
            {'if': {'column_id': 'SYM'}, 'textAlign': 'left',
             'fontWeight': '700', 'color': '#1a5c2a'},
            {'if': {'column_id': 'DIR'}, 'textAlign': 'center'},
            {'if': {'column_id': 'DIV'}, 'textAlign': 'center',
             'fontSize': '8px'},
        ],
        style_header={
            'backgroundColor': '#1a5c2a', 'color': '#fff',
            'fontWeight': '700', 'fontSize': '9px',
            'textAlign': 'center'},
        style_data_conditional=[
            {'if': {'filter_query': '{DIR} = "LONG"',
                    'column_id': 'DIR'},
             'color': '#1a8c3a', 'fontWeight': 'bold'},
            {'if': {'filter_query': '{DIR} = "SHORT"',
                    'column_id': 'DIR'},
             'color': '#cc2200', 'fontWeight': 'bold'},
            {'if': {'filter_query': '{STS} >= 10'},
             'backgroundColor': '#e8f5e9'},
            {'if': {'filter_query': '{STS} >= 7 && {STS} < 10'},
             'backgroundColor': '#f1f8e9'},
            {'if': {'column_id': 'PRICE'},
             'color': '#0055aa', 'fontWeight': '600'},
            {'if': {'column_id': 'TP'}, 'color': '#1a8c3a'},
            {'if': {'column_id': 'SL'}, 'color': '#cc2200'},
            {'if': {'filter_query': '{TMS} >= 35',
                    'column_id': 'TMS'},
             'backgroundColor': '#1a8c3a', 'color': '#fff'},
            {'if': {'filter_query': '{TMS} >= 25 && {TMS} < 35',
                    'column_id': 'TMS'},
             'backgroundColor': '#4caf50', 'color': '#fff'},
            {'if': {'filter_query': '{TMS} >= 15 && {TMS} < 25',
                    'column_id': 'TMS'},
             'backgroundColor': '#81c784', 'color': '#fff'},
            {'if': {'filter_query': '{CMS} >= 8',
                    'column_id': 'CMS'},
             'backgroundColor': '#e65100', 'color': '#fff'},
            {'if': {'filter_query': '{CMS} >= 5 && {CMS} < 8',
                    'column_id': 'CMS'},
             'backgroundColor': '#f57c00', 'color': '#fff'},
            {'if': {'filter_query': '{CMS} >= 3 && {CMS} < 5',
                    'column_id': 'CMS'},
             'backgroundColor': '#ffb74d', 'color': '#000'},
            {'if': {'filter_query': '{DIV} = "P>R"',
                    'column_id': 'DIV'},
             'backgroundColor': '#fff3e0', 'color': '#e65100'},
            {'if': {'filter_query': '{DIV} = "P<R"',
                    'column_id': 'DIV'},
             'backgroundColor': '#e8f5e9', 'color': '#1b5e20'},
            {'if': {'row_index': 'odd'},
             'backgroundColor': '#f0ebe0'},
        ],
        tooltip_header={
            'C@E': 'Correlation at last earnings date',
            'C@N': 'Correlation now (with live price)',
            'ΔCOR': 'Delta: C@N minus C@E (neg = decorrelating)',
            'DCOR': 'Decorrelation score (0-1)',
            'CMS': 'Correlation Merit Score',
            'DIV': 'Price vs Revenue divergence',
        },
    ),

    html.Div("© 2026 Truth Communications LLC • STASIS AM",
             className="text-center",
             style={'fontSize': '8px', 'color': '#888',
                    'padding': '4px'}),

], style={'background': '#f5f0e8', 'minHeight': '100vh'})


# ============================================================================
# CALLBACKS
# ============================================================================


@app.callback(
    Output('status', 'children'), Input('tick', 'n_intervals'))
def update_status(n):
    if not manager.backfill_complete:
        return html.Span(
            f"⏳ Initializing... {manager.backfill_progress}%",
            style={'color': '#aa6600'})
    st = price_feed.get_status()
    am_data = manager.get_am_data()
    tradable = sum(1 for d in am_data if d.get('is_tradable'))
    corr_count = sum(
        1 for v in config.correlation_data.values()
        if v.get('corr_at_earnings') is not None)
    decorr_count = sum(
        1 for v in config.correlation_data.values()
        if (v.get('decorrelation_score') or 0) > 0.3)
    if st['connected'] == 0:
        return html.Span(
            f"🔴 Connecting... | {tradable} tradable",
            style={'color': '#aa6600'})
    return html.Span(
        f"🟢 LIVE {st['connected']}/{st['total']} | "
        f"📨 {st['messages']:,} msgs | "
        f"📊 {len(config.fundamental_slopes)} fundamentals | "
        f"🔗 {corr_count} correlations "
        f"({decorr_count} decorrelating) | "
        f"🎯 {tradable} tradable",
        style={'color': '#1a5c2a'})


@app.callback(
    [Output('f-all', 'active'), Output('f-trad', 'active'),
     Output('f-decorr', 'active'), Output('fmode', 'data')],
    [Input('f-all', 'n_clicks'), Input('f-trad', 'n_clicks'),
     Input('f-decorr', 'n_clicks')],
    prevent_initial_call=True)
def toggle_filter(n1, n2, n3):
    ctx = callback_context
    tid = ctx.triggered[0]['prop_id']
    if 'f-all' in tid:
        return True, False, False, 'all'
    elif 'f-decorr' in tid:
        return False, False, True, 'decorr'
    return False, True, False, 'tradable'


@app.callback(
    Output('tbl', 'data'),
    [Input('tick', 'n_intervals'), Input('fmode', 'data'),
     Input('f-dir', 'value'), Input('f-sort', 'value')])
def update_table(n, fm, fd, fs):
    if not manager.backfill_complete:
        return []
    data = manager.get_am_data()
    if not data:
        return []
    rows = []
    for d in data:
        if fm == 'tradable' and not d.get('is_tradable'):
            continue
        if fm == 'decorr':
            cd = d.get('corr_details', {})
            dcor = cd.get('decor_score')
            if dcor is None or dcor < 0.15:
                continue
        if fd != 'ALL' and d.get('direction') != fd:
            continue

        sd = d.get('slope_details', {})
        cd = d.get('corr_details', {})
        w52 = d.get('week52_percentile')

        corr_at_e = cd.get('corr_at_earnings')
        corr_now = cd.get('corr_now')
        corr_delta = cd.get('corr_delta')
        decor_score = cd.get('decor_score')
        divergence = cd.get('divergence')

        div_display = '—'
        if divergence == 'PRICE_AHEAD':
            div_display = 'P>R'
        elif divergence == 'PRICE_BEHIND':
            div_display = 'P<R'
        elif divergence == 'ALIGNED':
            div_display = '≈'

        rows.append({
            '✓': '✅' if d.get('is_tradable') else '',
            'SYM': d['symbol'],
            'BAND': f"{d['threshold_pct']:.2f}%",
            'STS': d['stasis'],
            'DIR': d.get('direction') or '—',
            'SMS': d.get('sms', 0),
            'FMS': d.get('fms', 0),
            'CMS': d.get('cms', 0),
            'TMS': d.get('tms', 0),
            'C@E': fmt_corr(corr_at_e),
            'C@N': fmt_corr(corr_now),
            'ΔCOR': fmt_corr_delta(corr_delta),
            'DCOR': (f"{decor_score:.2f}"
                     if decor_score is not None else '—'),
            'DIV': div_display,
            'REV5': fmt_slope(sd.get('Rev_5')),
            'FCF5': fmt_slope(sd.get('FCF_5')),
            'FCFY': (f"{sd['FCFY'] * 100:.1f}%"
                     if sd.get('FCFY') else '—'),
            '52W': (f"{w52:.0f}%"
                    if w52 is not None else '—'),
            'PRICE': (f"${d['current_price']:.2f}"
                      if d.get('current_price') else '—'),
            'TP': (f"${d['take_profit']:.2f}"
                   if d.get('take_profit') else '—'),
            'SL': (f"${d['stop_loss']:.2f}"
                   if d.get('stop_loss') else '—'),
            'R:R': fmt_rr(d.get('risk_reward')),
            'DUR': d.get('stasis_duration_str', '—'),
            # hidden sort cols
            '_tms': d.get('tms', 0),
            '_fms': d.get('fms', 0),
            '_cms': d.get('cms', 0),
            '_stasis': d['stasis'],
            '_52w': w52 if w52 is not None else 999,
            '_dcor_raw': (decor_score
                          if decor_score is not None else -1),
            '_corr_delta': (corr_delta
                            if corr_delta is not None else 999),
        })
    if not rows:
        return []
    df = pd.DataFrame(rows)
    sort_map = {
        'tms': ('_tms', False),
        'fms': ('_fms', False),
        'cms': ('_cms', False),
        'decorr': ('_dcor_raw', False),
        'corr_delta': ('_corr_delta', True),
        'stasis': ('_stasis', False),
        '52w': ('_52w', True),
    }
    col, asc = sort_map.get(fs, ('_tms', False))
    df = df.sort_values(col, ascending=asc).head(200)
    df = df.drop(
        columns=['_tms', '_fms', '_cms', '_stasis', '_52w',
                 '_dcor_raw', '_corr_delta'],
        errors='ignore')
    return df.to_dict('records')


# ============================================================================
# API ENDPOINTS
# ============================================================================


@server.route('/api/health')
def health():
    return json.dumps({
        'status': 'ok', 'app': 'stasis_am',
        'initialized': manager.initialized,
        'backfill_complete': manager.backfill_complete,
        'backfill_progress': manager.backfill_progress,
        'correlations_calculated': sum(
            1 for v in config.correlation_data.values()
            if v.get('corr_at_earnings') is not None),
    })


@server.route('/api/correlations')
def api_correlations():
    result = {}
    for sym, data in config.correlation_data.items():
        if data.get('corr_at_earnings') is not None:
            result[sym] = {
                'corr_at_earnings': data['corr_at_earnings'],
                'corr_now': data.get('corr_now'),
                'corr_delta': data.get('corr_delta'),
                'decorrelation_score': data.get('decorrelation_score'),
                'divergence': data.get('price_vs_rev_divergence'),
                'earnings_date_price': data.get('earnings_date_price'),
                'current_price': data.get('current_price'),
                'price_change_since_earnings_pct': data.get(
                    'price_change_since_earnings_pct'),
                'latest_rev_change_pct': data.get(
                    'latest_rev_change_pct'),
                'aligned_quarters': data.get('aligned_count'),
            }
    return json.dumps(result, indent=2)


@server.route('/api/decorrelating')
def api_decorrelating():
    items = []
    for sym, data in config.correlation_data.items():
        if data.get('decorrelation_score') is not None:
            items.append({
                'symbol': sym,
                'decorrelation_score': data['decorrelation_score'],
                'corr_at_earnings': data.get('corr_at_earnings'),
                'corr_now': data.get('corr_now'),
                'corr_delta': data.get('corr_delta'),
                'divergence': data.get('price_vs_rev_divergence'),
            })
    items.sort(key=lambda x: x['decorrelation_score'], reverse=True)
    return json.dumps(items, indent=2)


# ============================================================================
# INITIALIZATION
# ============================================================================

_init_done = False
_init_lock = threading.Lock()


def initialize():
    global _init_done
    with _init_lock:
        if _init_done:
            return
        print("=" * 70)
        print("  STASIS AM SERVER")
        print("  © 2026 Truth Communications LLC")
        print("  Price:Revenue Correlation Analysis")
        print("=" * 70)
        print(f"\n🎯 Symbols: {len(config.symbols)}")

        config.week52_data = fetch_52_week_data()
        config.volumes = fetch_volume_data()
        fetch_all_fundamental_data()

        # Correlation — uses its own self-contained pipeline
        calculate_all_correlations()

        manager.backfill()
        price_feed.start()
        manager.start()

        corr_ok = sum(
            1 for v in config.correlation_data.values()
            if v.get('corr_at_earnings') is not None)
        decorr = sum(
            1 for v in config.correlation_data.values()
            if (v.get('decorrelation_score') or 0) > 0.3)

        print(f"\n✅ READY")
        print(f"   📊 {len(config.fundamental_slopes)} fundamentals")
        print(f"   🔗 {corr_ok} correlations calculated")
        print(f"   🔔 {decorr} significantly decorrelating")
        print("=" * 70)
        _init_done = True


_init_thread = threading.Thread(target=initialize, daemon=True)
_init_thread.start()

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    _init_thread.join()
    port = int(os.environ.get('PORT', 8050))
    print(f"\n🟢 http://0.0.0.0:{port}\n")
    app.run(debug=False, host='0.0.0.0', port=port)
