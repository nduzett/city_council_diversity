import pandas as pd
import copy
#import time

in_file_cc = 'C:\\Users\\Neil\\Desktop\\gh_city_councils.csv'
in_file_cl = 'C:\\Users\\Neil\\Desktop\\gh_county_legislatures.csv'
in_file_sb = 'C:\\Users\\Neil\\Desktop\\gh_school_boards.csv'
df = pd.read_csv(in_file_sb)

df = df[['ledb_candid', 'geo_name', 'state_abb', 'year', 'district', 'contest', 'n_winners', 'winner']]

df['district'] = df['district'].str.replace(' runoff needed', '')
df['district'] = df['district'].str.replace(' run-off needed', '')
df['district'] = df['district'].str.replace(' needs runoff', '')
df['district'] = df['district'].str.replace(' needs run off', '')
df['district'] = df['district'].str.replace(' runoff', '')
df['district'] = df['district'].str.replace(' run-off', '')
df['district'] = df['district'].str.replace('*', '')
df['district'] = df['district'].str.replace(' unexpired', '')
df['district'] = df['district'].str.replace('superward ', '')
df['district'] = df['district'].str.replace('at large', 'at-large')

# identify cities without at-large elections

def get_al_elec(district):
    if "at-large" in district:
        return 1
    else:
        return 0

df['district'] = df['district'].astype(str)
df['al_elec'] = df['district'].apply(get_al_elec)
df['city_has_al_elec'] = df.groupby(['state_abb', 'geo_name'])['al_elec'].transform('sum')
df = df.drop(df[df['city_has_al_elec'] == 0].index)
df = df.drop(df[df['winner'] != "win"].index)
df = df.drop(['al_elec', 'city_has_al_elec', 'winner'], axis = 1)
df = df.sort_values(by=['state_abb', 'geo_name', 'year', 'district'])

# within each city, create a dictionary for each year with keys for district names and values for candidate ids

"""
df = df[df['geo_name'] == 'santa cruz']
df = df[df['geo_name'] == 'alameda']
"""

city_grouped = df.groupby(['state_abb', 'geo_name'])

city_councils = {}
city_councils_deep = {}
city_year_full = {}

# put together who is on the council each year, compute distribution of filled seats over the years, 
#   identify where it stabilizes, throw out the years pre stabilization

# take one city at a time
for city in city_grouped:
    city_df = city[1]
    city_df = city_df.sort_values(by = ['year'])
    year_grouped = city_df.groupby(['year'])
    
    state_city_year_key_old = city[0] + tuple([1000])
    city_councils[state_city_year_key_old] = {}
    
    # each year, update the council in that city
    for year in year_grouped:
        year_df = year[1]
        """
        print("")
        print(year[0][0])
        """
        
        # copy over last year's council
        state_city_year_key = city[0] + year[0]
        city_councils[state_city_year_key] = copy.deepcopy(city_councils[state_city_year_key_old])
        
        # identify which districts will be updated this year
        districts_to_update = year_df['district'].unique()
        
        # remove all existing councilors from those districts
        for district in districts_to_update:
            city_councils[state_city_year_key][district] = ("", year[0][0])
        
        # add the newly elected councilors
        for index, row in year_df.iterrows():
            state_abb = row['state_abb']
            city_name = row['geo_name']
            
            # add semicolon as delimiter
            if city_councils[state_city_year_key][row['district']][0] != "":
                city_councils[state_city_year_key][row['district']] = (city_councils[state_city_year_key][row['district']][0] + ";", year[0][0])
            
            # add the councilor
            city_councils[state_city_year_key][row['district']] = (city_councils[state_city_year_key][row['district']][0] + str(row['ledb_candid']), year[0][0])
            """
            time.sleep(2)
            print("District: {}".format(row['district']))
            print("{}".format(city_councils[state_city_year_key][row['district']]))
            """
            
        # delete districts that haven't been updated within the last few cycles
        districts_to_delete = []
        for district in city_councils[state_city_year_key].keys():
            if city_councils[state_city_year_key][district][1] + 4 <= year[0][0]:
                districts_to_delete.append(district)
        
        for district in districts_to_delete:
            city_councils[state_city_year_key].pop(district)
        
        # setup this year's council to copy over for next year
        state_city_year_key_old = state_city_year_key
        """
        print(city_councils[state_city_year_key_old])
        """
        
    # determine at which year the city council becomes "full"
    years = city_df['year'].unique()
    seats_by_year = []
    
    print("")
    print("{}, {}".format(city_name, state_abb))
    
    # creates a list of the number of seats in each year
    for year in years:
        state_city_year_key = city[0] + tuple([year])
        num_seats = 0
        
        for key in city_councils[state_city_year_key].keys():
            #print(city_councils[state_city_year_key][key])
            num_seats = num_seats + city_councils[state_city_year_key][key][0].count(';') + 1
        
        seats_by_year.append(num_seats)
        
        #print("{}: {} seats in {} districts".format(year, num_seats, len(city_councils[state_city_year_key].keys())))
        
        
    # examine the list of seats and determine when the number of seats becomes stable
    year_full = 0
    for x in range(len(seats_by_year)):
        # not the last element, 
        if x != len(seats_by_year) - 1 and x != len(seats_by_year) - 2:
            # if same number of seats three election years in a row, then the council is full
            if seats_by_year[x] == seats_by_year[x + 1] == seats_by_year[x + 2]:
                year_full = years[x]
                break
    
    city_year_full[city_name] = year_full
    
    #print("Predicted year full: {}".format(year_full))
    #time.sleep(15)
    

"""
print(city_councils)
"""
      
city_councils_flat = {}

for city_council_key in city_councils.keys():
    city_council_string = ""
    counter = 1
    
    for seat_key in city_councils[city_council_key]:
        if counter != 1:
            city_council_string = city_council_string + ";"
        city_council_string = city_council_string + str(seat_key) + "_" + str(city_councils[city_council_key][seat_key])
        counter += 1
        
    city_councils_flat[city_council_key] = city_council_string
    
councils_df = pd.concat({k: pd.DataFrame.from_dict(v, 'index') for k, v in city_councils.items()}, axis=0)
councils_df.to_csv("C:\\Users\\Neil\\Desktop\\gh_sitting_sb.csv", header = False)

councils_year_full_df = pd.DataFrame.from_dict(city_year_full, 'index')
councils_year_full_df.to_csv("C:\\Users\\Neil\\Desktop\\gh_sitting_sb_year_full.csv", header = False)




