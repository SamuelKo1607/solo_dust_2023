import requests
import os
import datetime
import cdflib

def download(YYYYMMDD,directory,product_family,product,version_priority): #will download rpw tds from lesia
    #product example: _rpw-tds-surv-tswf-e_
    #product example: _rpw-tds-surv-stat_
    #product example: _rpw-bia-scpot-10-seconds_
    
    #product family example: tds_stat
    #product family example: tds_wf_e
    #product family example: 
    
    #version priority exapmle: ["V02.cdf","V01.cdf"]
    
    print("***** "+YYYYMMDD + ": Download initiated *****")
    a = datetime.datetime.now()
    os.chdir(directory)
    date=str(YYYYMMDD) 
    if product_family == "lfr_scpot":
        myurl = "https://rpw.lesia.obspm.fr/roc/data/pub/solo/rpw/data/L3/"+product_family+"/"
    else:
        myurl = "https://rpw.lesia.obspm.fr/roc/data/pub/solo/rpw/data/L2/"+product_family+"/"
    year = date[:4]
    month = date[4:6]
    
    success = False
    
    for version in version_priority:
        if not success:
            suffix = "_"+version
            if product_family == "lfr_scpot":
                short_file = "solo_L3"+product+date+suffix
            else:
                short_file = "solo_L2"+product+date+suffix
            myfile = myurl+year+"/"+month+"/"+short_file
            r = requests.get(myfile, allow_redirects=True)
            if not str(r)=="<Response [404]>":
                success = True
                open(short_file, 'wb').write(r.content)
                print("***** "+YYYYMMDD + short_file+" downlaoded")
                print("***** "+str(round(os.path.getsize(short_file)/(1024**2),ndigits=2))+" MiB dowloaded in "+str(datetime.datetime.now()-a))
            else:
                print("***** "+YYYYMMDD + ": *_"+version+" not available")
    if success:
        print("***** "+YYYYMMDD + ": Download success *****")
        return short_file
    else:
        print("***** "+YYYYMMDD + ": Download failure *****")
        return "N/A"



def fetch(YYYYMMDD,directory,productfamily,product,version_priority,access_online): #will access local data or download if needed
    os.chdir(directory)
    short_file = "N/A"
    for version in version_priority:
        try:
            if productfamily == "lfr_scpot":
                f = open("solo_L3"+product+str(YYYYMMDD)+"_"+version)
            else:
                f = open("solo_L2"+product+str(YYYYMMDD)+"_"+version)
        except:
            pass
            #print("cant open "+"solo_L2"+product+str(YYYYMMDD)+"_"+version)
        else:
            f.close()
            if productfamily == "lfr_scpot":
                name = "solo_L3"+product+str(YYYYMMDD)+"_"+version
            else:
                name = "solo_L2"+product+str(YYYYMMDD)+"_"+version   
            cdf_file = cdflib.CDF(name)
            print(str(YYYYMMDD)+": loaded locally as: "+name)
            return cdf_file
    #only if nothing was returned
    if short_file == "N/A":
        if access_online:
            short_file = download(YYYYMMDD,directory,productfamily,product,version_priority)
    if short_file == "N/A":
        raise LookupError 
    else:
        cdf_file = cdflib.CDF(short_file)
        print(YYYYMMDD + ": Fetched "+short_file)
        return cdf_file
    

 
