import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import streamlit as st
from matplotlib.ticker import FuncFormatter


# List of US state names and abbreviations
us_states = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA",
    "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
    "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
    "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
    "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"
}


### User inputs #####

# Function to calculate the percentage eligible
def calculate_percentage_eligible(num_employees, hourly_pct, hr_min, hr_median, hr_max, avg_dep, state):
    ### General assumptions ####

    avg_weekly_hrs = 40 # average hours worked per week

    weeks_in_yrs = 52 # weeks employees are expected to work each year



    # User-defined distributions for household size and parental status
    distribution_percentages = {
     1: 0.289, 2: 0.347, 3: 0.151, 4: 0.123, 5: 0.056, 6: 0.034
    }

    percentages = {
        "Single": 0.464,   # employee without partner or child
        "Dual": 0.3,    # employee with partner and no child
        "Single_P": 0.041, # employee with child and no partner
        "Dual_P": 0.195     # employee with both partner and child
    }

    # Assumption: Percentage of married with both employed (source: BLS)
    married_both_employed_percentage = 0.497


    ############################
    #### Generating Salaries ####
    ############################
    np.random.seed(42)  # for reproducibility

    num_samples = num_employees - 3  # Number of samples, leaving space for one min, one median, and one max value


    # Step 1: Simulate using the lognormal distribution with the standard deviation to be 2.5% of the mean
    sigma_initial = 0.025 * hr_median   # Estimate for hourly wage from bls.gov

    # Initialize hourly pay data
    hourly_pay_data = np.random.lognormal(mean=np.log(hr_median), sigma=sigma_initial, size=num_samples)

    # Ensure at least one simulated entry has the minimum value of 10, median value of 20, and maximum value of 40
    hourly_pay_data = np.append(hourly_pay_data, [hr_min, hr_median, hr_max])

    # Step 2 and 3: Resimulate data falling outside the min-max interval until all values are within the range
    while not np.all((hourly_pay_data >= hr_min) & (hourly_pay_data <= hr_max)):
        # Find indices where values fall outside the range
        outside_indices = np.logical_or(hourly_pay_data < hr_min, hourly_pay_data > hr_max)

        # Resimulate values outside the range using the same median but half the standard deviation
        hourly_pay_data[outside_indices] = np.random.lognormal(mean=np.log(hr_median), sigma=sigma_initial/2, size=np.sum(outside_indices))

    # Print the hourly pay data
    # print(hourly_pay_data)


    # Round the salary to two decimal places
    salary_data = np.round(hourly_pay_data, 2)



    # Create a DataFrame with 'id' and 'salary' columns
    data = {
        'id': range(1, num_employees + 1),  # Generate employee IDs from 1 to max
        'hourly_salary': salary_data
    }

    # Create the DataFrame
    df_salary_initial = pd.DataFrame(data)


    ## Check ##
    below_thres = np.sum(df_salary_initial['hourly_salary'] < hr_median)
    #print(below_thres)


    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df_salary_initial['hourly_salary'], bins=20, color='skyblue', edgecolor='black')  # Adjust bins as needed
    plt.title('Hourly Pay Distribution')
    plt.xlabel('Salary')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    #fig, ax = plt.subplots()
    #ax.hist(df_salary['hourly_salary'], bins=20, color='skyblue', edgecolor='black')
    #st.pyplot(fig)

    df = df_salary_initial

    # Calculate annual salary and create a new column 'annual_salary'
    df['annual_salary'] = df['hourly_salary'] * avg_weekly_hrs * weeks_in_yrs

    df['annual_salary'] = df['annual_salary'].apply(lambda x: math.floor(x))

    #print(df)




    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df['annual_salary'], bins=20, color='skyblue', edgecolor='black')  # Adjust bins as needed
    plt.title('Annual Salary Distribution')
    plt.xlabel('Salary')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


    # Create the "State/Province" column with all rows labeled with the assigned value
    df['State/Province'] = state


    ############################
    ### Generating Household Size
    ############################ 
    np.random.seed(42)  # for reproducibility

    # Calculate cumulative probabilities
    cumulative_prob = {}
    cumulative = 0
    for size, percentage in distribution_percentages.items():
        cumulative += percentage
        cumulative_prob[size] = cumulative

    # Generate household size data for specified individuals
    household_size_data = []
    for _ in range(num_employees):
        rand_num = np.random.rand()  # Generate random number between 0 and 1

        # Assign the individual to a household size category based on cumulative probabilities
        for size, prob in cumulative_prob.items():
            if rand_num <= prob:
                household_size_data.append(size)
                break

    # Print the generated household size data
    # print(household_size_data)


    # Create a column for household_size
    df['household_size'] = household_size_data



    ## Checks ##
    # Display the distribution of categories
    household_dist = df['household_size'].value_counts(normalize=True) * 100
    #print("Distribution of Household Size:")
    #print(household_dist.sort_index())


    plt.figure(figsize=(10, 6))
    df['household_size'].hist(bins=range(1, len(distribution_percentages) + 2), align='left', color='skyblue', edgecolor='black')
    plt.title('Household Size Distribution in Frequency Count')
    plt.xlabel('Size')
    plt.ylabel('Frequency')
    plt.xticks(range(1, len(distribution_percentages) + 1))
    plt.grid(axis='y')
    plt.show()


    # Calculate weights for each entry to convert counts to percentages
    weights = np.ones_like(df['household_size']) / len(df) * 100

    plt.figure(figsize=(10, 6))
    df['household_size'].hist(bins=range(1, len(distribution_percentages) + 2), align='left', color='skyblue', edgecolor='black', weights=weights)
    plt.title('Household Size Distribution in Percentages')
    plt.xlabel('Size')
    plt.ylabel('Percentage (%)')
    plt.xticks(range(1, len(distribution_percentages) + 1))
    plt.grid(axis='y')
    plt.show()




    ###########################
    # Generating Parental Status
    ############################
    np.random.seed(42)  # for reproducibility
    
    # Count the number of employees with household size = 1
    num_single_household = df[df['household_size'] == 1].shape[0]

    #print("Number of single households:", num_single_household)


    # Adjust the distribution percentages for household sizes greater than 1
    adjusted_single_percentage = percentages["Single"] * (1 - num_single_household / num_employees)
    adjusted_percentages = {
        "Single": adjusted_single_percentage,
        "Dual": percentages["Dual"],
        "Single_P": percentages["Single_P"],
        "Dual_P": percentages["Dual_P"]
    }

    #print("Adjusted single percentage:", adjusted_single_percentage)

    # print(adjusted_percentages)


    # Calculate the total adjusted percentage
    total_adjusted_percentage = sum(adjusted_percentages.values())

    # print(total_adjusted_percentage)

    # Calculate the remaining percentage to reach 1
    remaining_percentage = 1.0 - total_adjusted_percentage

    # print(remaining_percentage)

    # print(remaining_percentage+total_adjusted_percentage)



    # Adjust the percentages for categories other than "Single" based on the original distribution
    for category in adjusted_percentages:
        if category != "Single":
            percentage = percentages[category]
            adjusted_percentages[category] += remaining_percentage * (percentage / (1 - percentages["Single"]))
            
            
            

    # Ensure the total adjusted percentages sum up to 1
    total_adjusted_percentage = sum(adjusted_percentages.values())



    # Display the adjusted percentages
    #print("Adjusted Percentages:", adjusted_percentages)
    #print("Total Adjusted Percentage:", total_adjusted_percentage)


    # Assign parental status based on household size and the adjusted distribution
    for index, row in df.iterrows():
        household_size = row['household_size']
        if household_size == 1:
            df.at[index, 'parental_status'] = 'Single'
        else:
            rand = np.random.rand()  # Generate a random number between 0 and 1
            cumulative_prob = 0
            for status, percentage in adjusted_percentages.items():
                cumulative_prob += percentage
                if rand <= cumulative_prob:
                    df.at[index, 'parental_status'] = status
                    break

    # Display the distribution of categories
    distribution = df['parental_status'].value_counts(normalize=True) * 100
    #print("Distribution of Parental Status:")
    #print(distribution)



    # Group by household size and parental status, count occurrences, and unstack
    grouped = df.groupby(['household_size', 'parental_status']).size().unstack(fill_value=0)

    #print(grouped)



    grouped_pct = grouped.div(grouped.sum(axis=1), axis=0) * 100  # Calculate percentages

    grouped_pct = grouped_pct.applymap(lambda x: '{:.2f}'.format(x))

    #print(grouped_pct)

    # Calculate the total percentage for each parental status category across all household sizes
    total_percentage = grouped.sum(axis=0)/grouped.sum().sum() * 100   ## Divided by the total number of employees 
    #print(total_percentage)



    # Count the occurrences of each category in 'parental_status'
    counts = df['parental_status'].value_counts()

    # Plotting
    plt.figure(figsize=(10, 6))
    counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Parental Status Distribution in Frequency Count')
    plt.xlabel('Parental Status')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)  # Rotate labels to make them readable
    plt.grid(axis='y')
    plt.show()



    # Calculate the percentage of each category in 'parental_status'
    parental_pct = df['parental_status'].value_counts(normalize=True) * 100

    # Plotting
    plt.figure(figsize=(10, 6))
    parental_pct.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Parental Status Distribution in Percentages')
    plt.xlabel('Parental Status')
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=45)  # Rotate labels to make them readable
    plt.grid(axis='y')
    plt.show()



    ################################
    ### Calculations ###
    ##############################

    ########################################
    ###### Define Marital status
    #######################################
    df['Marital Status'] = np.where(df['parental_status'].isin(['Dual', 'Dual_P']), 'M', 'S')

    ################################
    ### Calculations ###
    ##############################
    # Set random seed for reproducibility
    np.random.seed(42)


    # Get indices of 'M' rows
    m_indices = df[df['Marital Status'] == 'M'].index.to_numpy()

    # Randomly shuffle the indices
    np.random.shuffle(m_indices)

    # Determine the number of 'BE' and 'SE' based on the percentage assumption
    num_be = int(len(m_indices) * married_both_employed_percentage)


    # Assign 'BE' and 'SE' labels to 'M' rows
    df.loc[m_indices[:num_be], 'M- Employment Type'] = 'BE'  #BE is both employed in Married 
    df.loc[m_indices[num_be:], 'M- Employment Type'] = 'SE'  #SE is only one employed in Married

    # Summary of 'M- Employment Type'
    summary = df['M- Employment Type'].value_counts(normalize=True) * 100
    # print(summary)

    # Calculate total household salary based on employment type
    df['total_household_income'] = df['annual_salary']
    df.loc[df['M- Employment Type'] == 'BE', 'total_household_income'] *= 2


    # Function to calculate child status based on parental status
    def calculate_child_status(parental_status):
        if parental_status in ['Dual_P', 'Single_P']:
            return 1
        else:
            return 0

    # Apply the function to create the 'child_status' column
    df['child_status'] = df['parental_status'].apply(lambda x: calculate_child_status(x))

    # Display the updated DataFrame
    #print(df)


    ################################
    ###### Creating dataframe for salary summary ############
    ################################
    # Define salary bins
    salary_bins = [0, 20000, 40000, 60000, 80000, 100000, np.inf]

    # Create salary bands using cut
    df['Salary Band'] = pd.cut(df['annual_salary'], bins=salary_bins, right=False)

    # Count salaries in each band
    salary_counts = df['Salary Band'].value_counts().sort_index()

    # Calculate total salaries
    total_salaries = len(df)

    # Calculate percentage of total salaries in each band
    percentage_salary = (salary_counts / total_salaries) * 100

    # Create DataFrame
    df_salary = pd.DataFrame({
        'Salary Band': salary_counts.index,
        'Count': salary_counts.values,
        'Percentage of Total': percentage_salary.values
    })

    # Define the mapping
    salary_band_mapping = {
        pd.Interval(left=0, right=20000, closed='left'): '$0 - $20K',
        pd.Interval(left=20000, right=40000, closed='left'): '$20K - $40K',
        pd.Interval(left=40000, right=60000, closed='left'): '$40K - $60K',
        pd.Interval(left=60000, right=80000, closed='left'): '$60K - $80K',
        pd.Interval(left=80000, right=100000, closed='left'): '$80K - $100K',
        pd.Interval(left=100000, right=np.inf, closed='left'): '$100K+'
    }

    # Replace the 'Salary Band' values using the mapping
    df_salary['Salary Band'] = df_salary['Salary Band'].map(salary_band_mapping)

    print(df_salary)

    # print(df_salary['Count'].sum())
    # print(df.columns)

    ####################################################
    # Create household income bands using cut
    df['Salary Band HI'] = pd.cut(df['total_household_income'], bins=salary_bins, right=False)

    # Count salaries in each band
    income_counts = df['Salary Band HI'].value_counts().sort_index()

    # Calculate total salaries
    total_incomes = len(df)

    # Calculate percentage of total salaries in each band
    percentage_income = (income_counts / total_incomes) * 100

    # Create DataFrame
    df_household_income = pd.DataFrame({
        'Salary Band HI': income_counts.index,
        'Count': income_counts.values,
        'Percentage of Total': percentage_income.values
    })


    # Replace the 'Salary Band' values using the mapping
    df_household_income['Salary Band HI'] = df_household_income['Salary Band HI'].map(salary_band_mapping)


    print(df_household_income)

    #print(df)


    # Function to calculate child status based on parental status
    def calculate_child_status(parental_status):
        if parental_status in ['Dual_P', 'Single_P']:
            return 1
        else:
            return 0

    # Apply the function to create the 'child_status' column
    df['child_status'] = df['parental_status'].apply(lambda x: calculate_child_status(x))

    # Display the updated DataFrame
    #print(df)


    ###########################
    ### Dataset for FPL, SMI and matching to employees ###
    ############################

    #Creating a dataset for the FPL

    FPL = {
        'household_size': [1, 2, 3, 4, 5, 6],
        'FPL': [14580, 19720, 24860, 30000, 35140, 40280]
    }

    # Creating the DataFrame
    df_FPL = pd.DataFrame(FPL)

    # Displaying the DataFrame
    #print(df_FPL)


    # Create a dictionary to map household_size to FPL
    FPL_mapping = dict(zip(df_FPL['household_size'], df_FPL['FPL']))

    # Map household_size to FPL using the dictionary
    df['FPL'] = df['household_size'].map(FPL_mapping)



    # df.to_excel("WIP_FPL.xlsx", index=False)
    # df.to_csv("WIP_FPL.csv", index=False)


    #################################################
    ###### Index Match for SMI ############
    ##################################################

    # Import dataset

    file_path_inputs = 'Inputs.xlsx'

    # Specify the sheet name you want to load
    sheet_name = 'SMI'

    # Read the .xlsx file into a DataFrame
    df_SMI = pd.read_excel(file_path_inputs, sheet_name=sheet_name)
    df_SMI.head()



    #######################################################
    #### match SMI values for state and household-size #####

    # Create a dictionary to map the household size values from df_SMI
    smi_dict = df_SMI.set_index('State/Province').to_dict(orient='index')

    # Define a function to get the SMI value based on State and household_size
    def get_smi(row):
        state = row['State/Province']
        household_size = row['household_size']  # Convert to string to match column names in df_SMI
        if state in smi_dict and household_size in smi_dict[state]:
            return smi_dict[state][household_size]
        else:
            return None  # Return None if state or household_size not found in df_SMI

    # Apply the function to create the 'SMI' column in df
    df['SMI'] = df.apply(get_smi, axis=1)

    #print(df)


    # df.to_excel("WIP_Eligibility.xlsx", index=False)





    ###########################################
    ######## Adjusting SMI & FPL #############
    #########################################

    # Import dataset

    file_path_inputs = 'Inputs.xlsx'

    # Specify the sheet name you want to load
    sheet_name = 'FPL_SMI'

    # Read the .xlsx file into a DataFrame
    df_IncAdj = pd.read_excel(file_path_inputs, sheet_name=sheet_name)
    #df_IncAdj.head()


    #df_IncAdj.columns

    ################################
    ## FPL Adjustment Factor #####

    # Create a dictionary from where keys are "State/Province" and values are "FPL"
    adj_fpl_dict = df_IncAdj.set_index('State/Province')['FPL'].to_dict()
    #print(adj_fpl_dict)

    # Map the counts from smi_dict to df based on "State/Province" column
    df['FPL_Fct'] = df['State/Province'].map(adj_fpl_dict)  ## gives the multiplication factor for FPL
    #print(df)


    ################################
    ## SMI Adjustment Factor #####

    # Create a dictionary from where keys are "State/Province" and values are "FPL"
    adj_smi_dict = df_IncAdj.set_index('State/Province')['SMI'].to_dict()
    #print(adj_smi_dict)

    # Map the counts from smi_dict to df based on "State/Province" column
    df['SMI_Fct'] = df['State/Province'].map(adj_smi_dict)  ## gives the multiplication factor for FPL
    #print(df)



    ##################################
    ### Adjusted FPL and SMI ####

    df['FPL_Adj'] = df['FPL'] * df['FPL_Fct']

    df['SMI_Adj'] = df['SMI'] * df['SMI_Fct']



    ##############################
    ### Eligibility Checks ###
    ##############################

    # FPL
    ######################

    # Create the FPL_check column then check if household income is less than the adjusted FPL
    df['FPL_check'] = (df['total_household_income'] <= df['FPL_Adj']).astype(int)

    # Filter rows where 'FPL_check' == 1
    FPL_flt = df[df['FPL_check'] == 1]
    #print(FPL_flt)
    #print(FPL_flt['FPL_check'].sum())       




    # SMI
    ######################

    # Create the FPL_check column then check if household income is less than the adjusted FPL
    df['SMI_check'] = (df['total_household_income'] <= df['SMI_Adj']).astype(int)

    # Filter rows where 'SMI_check' == 1
    SMI_flt = df[df['SMI_check'] == 1]
    #print(SMI_flt)
    #print(SMI_flt['SMI_check'].sum())     




    # Create Eligibility column for FPL based on having children
    ######################

    # Create a new column 'FPL_elg' based on conditions
    df['FPL_elg'] = (df['child_status'] == 1) & (df['FPL_check'] == 1)

    # Convert boolean values to 0 or 1
    df['FPL_elg'] = df['FPL_elg'].astype(int)

    # Filter rows where 'SMI_check' == 1
    FPL_elg_flt = df[df['FPL_elg'] == 1]

    #print(FPL_elg_flt['FPL_elg'].sum())
    #print(FPL_elg_flt['child_status'].sum())     




    # Create Eligibility column for SMI based on having children
    ######################

    # Create a new column 'SMI_elg' based on conditions
    df['SMI_elg'] = (df['child_status'] == 1) & (df['SMI_check'] == 1)

    # Convert boolean values to 0 or 1
    df['SMI_elg'] = df['SMI_elg'].astype(int)

    # Filter rows where 'SMI_check' == 1
    SMI_elg_flt = df[df['SMI_elg'] == 1]

    #print(SMI_elg_flt['SMI_elg'].sum())
    #print(SMI_elg_flt['child_status'].sum())     





    ################################
    ## List benefits available by state

    ################################
    ## FPL Based Benefits  #####

    # Create a dictionary from where keys are "State/Province" and values are "FPL_B"
    fpl_ben_dict = df_IncAdj.set_index('State/Province')['FPL_B'].to_dict()
    #print(fpl_ben_dict)

    # Map the counts from smi_dict to df based on "State/Province" column
    df['FPL_ben'] = df['State/Province'].map(fpl_ben_dict)  ## gives the multiplication factor for FPL
    #print(df)


    ################################
    ## SMI Based Benefits  #####

    # Create a dictionary from where keys are "State/Province" and values are "FPL_B"
    smi_ben_dict = df_IncAdj.set_index('State/Province')['SMI_B'].to_dict()
    #print(smi_ben_dict)

    # Map the counts from smi_dict to df based on "State/Province" column
    df['SMI_ben'] = df['State/Province'].map(smi_ben_dict)  ## gives the multiplication factor for FPL
    #print(df)


    ################################
    ## List eligible benefits available by state
    ################################

    ## FPL Based Benefits  #####

    # Create a new column 'fpl_elg_ben' based on conditions
    df['fpl_elg_ben'] = np.where(df['FPL_elg'] == 1, df['FPL_ben'], '')


    ## SMI Based Benefits  #####

    # Create a new column 'fpl_elg_ben' based on conditions
    df['smi_elg_ben'] = np.where(df['SMI_elg'] == 1, df['SMI_ben'], '')




    ################################
    ## List all eligible benefits available
    ################################

    # Create a new column 'Eligible Benefits' based on conditions
    df['Eligible Benefits'] = np.where(
        (df['fpl_elg_ben'] == '') & (df['smi_elg_ben'] == ''), 
        '',  # Case 1
        np.where(
            (df['fpl_elg_ben'] != '') & (df['smi_elg_ben'] == ''), 
            df['fpl_elg_ben'],  # Case 2
            np.where(
                (df['fpl_elg_ben'] == '') & (df['smi_elg_ben'] != ''), 
                df['smi_elg_ben'],  # Case 3
                df['fpl_elg_ben'] + ', ' + df['smi_elg_ben']  # Case 4
            )
        )
    )


    # Filter rows where "Eligible Benefits" isn't blank
    #eligible_benefits_rows = df[df['Eligible Benefits'] != '']



    #########################################################

    #### Final Output ############

    ###########################################################

    # Define a function to determine the value of "Child Care Eligible?"
    def determine_child_care_eligibility(row):
        if row['Eligible Benefits'] == '':
            return 'No'
        else:
            return 'Yes'

    # Apply the function to create the "Child Care Eligible?" column
    df['Child Care Eligible?'] = df.apply(determine_child_care_eligibility, axis=1)



    # Count the total "Yes" values to determine total eligibility
    eligible_count = df['Child Care Eligible?'].value_counts()['Yes']
    #print("Total Eligible for Child Care:", eligible_count)

    # Calculate the percentage of eligible employees
    percentage_eligible = (eligible_count / num_employees) * 100
    #print("Percentage of eligible employees:", "{:.2f}%".format(percentage_eligible))

    # Store the DataFrames in session state
    st.session_state.df = df
    st.session_state.df_salary = df_salary
    st.session_state.df_household_income = df_household_income
    
    # Use the input variables in the model's logic to calculate the final outcome
    # Here, just return a dummy percentage for demonstration
    return eligible_count, percentage_eligible  # Placeholder percentage

# Function to format the x-axis labels
def currency_formatter(x, pos):
    return f'${x/1000:.0f}K'

# Streamlit app layout
# Using markdown for title with adjusted size
#st.markdown("## ChildCare Subsidy Eligibility Calculator")


# Streamlit application
st.title("ChildCare Subsidy Eligibility")

# Organize inputs side by side using columns
# First row of inputs
col1, col2, col3 = st.columns(3)
with col1:
    num_employees = st.number_input("Number of Employees", min_value=1, value=1000)
with col2:
    avg_dep = st.number_input("Average Dependents", value=3)
with col3:
    state_full_name = st.selectbox("State", list(us_states.keys()), index=list(us_states.keys()).index("California"))  # Dropdown menu for US states with default value

# Convert state full name to abbreviation
state = us_states[state_full_name]

# Second row of inputs
col4, col5, col6 = st.columns(3)
with col4:
    hr_min = st.number_input("Minimum Hourly Rate", value=10)
with col5:
    hr_median = st.number_input("Median Hourly Rate", value=20)
with col6:
    hr_max = st.number_input("Maximum Hourly Rate", value=40)

# Keep hourly_pct slider separate since it spans full width
hourly_pct = st.slider("Percentage of Hourly Employees", 0.0, 1.0, 0.8)

# Inject custom CSS for tab styling
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab"] {
        font-size: 18px;
        font-weight: bold;
        border: 1px solid #ccc;
        border-radius: 5px;
        margin-right: 8px;
        padding: 10px 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# Create tabs with custom CSS for styling
tab1, tab2 = st.tabs(["Eligibility Calculation", "Checks"])

with tab1:
    st.header("Eligibility Calculation")
    # Calculate button
    if st.button("Calculate Eligibility"):
        eligible_count, percentage_eligible = calculate_percentage_eligible(
            num_employees, hourly_pct, hr_min, hr_median, hr_max, avg_dep, state
        )
        # Enhanced display using st.metric
        st.markdown(f"""
            <div style='text-align: center; color: black;'>
                <span style='font-size: x-large; font-weight: bold;'>
                    <span style='border-bottom: 3px double; padding-bottom: 2px;'>
                        {eligible_count} ({percentage_eligible:.1f}%)
                    </span> Employees Eligible
                </span>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.header("Checks")
    if 'df' in st.session_state and 'df_salary' in st.session_state and 'df_household_income' in st.session_state:
        df = st.session_state.df
        df_salary = st.session_state.df_salary
        df_household_income = st.session_state.df_household_income
        distribution_percentages = {1: 0.289, 2: 0.347, 3: 0.151, 4: 0.123, 5: 0.056, 6: 0.034}

        # Display DataFrames side by side
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Salary Band Distribution**")
            st.dataframe(df_salary)
        with col2:
            st.markdown("**Household Income Distribution**")
            st.dataframe(df_household_income)

        # Arrange plots in a grid with a maximum of two plots per row
        col1, col2 = st.columns(2)

        # Plot: Annual Salary Distribution
        with col1:
            plt.figure(figsize=(5, 4))
            plt.hist(df['annual_salary'], bins=20, color='skyblue', edgecolor='black')  # Adjust bins as needed
            plt.title('Annual Salary Distribution')
            plt.xlabel('Salary')
            plt.ylabel('Frequency')
            plt.gca().xaxis.set_major_formatter(FuncFormatter(currency_formatter))
            plt.grid(True)
            st.pyplot(plt)

        # Plot: Household Income Distribution
        with col2:
            plt.figure(figsize=(5, 4))
            # Assuming df has 'total_household_income' column, if not replace with correct column
            plt.hist(df['total_household_income'], bins=20, color='skyblue', edgecolor='black')  # Adjust bins as needed
            plt.title('Household Income Distribution')
            plt.xlabel('Income')
            plt.ylabel('Frequency')
            plt.gca().xaxis.set_major_formatter(FuncFormatter(currency_formatter))
            plt.grid(True)
            st.pyplot(plt)

        # Second row of plots
        col3, col4 = st.columns(2)

        # Plot: Household Size Distribution
        with col3:
            weights = np.ones_like(df['household_size']) / len(df) * 100
            plt.figure(figsize=(5, 4))
            df['household_size'].hist(bins=range(1, len(distribution_percentages) + 2), align='left', color='skyblue', edgecolor='black', weights=weights)
            plt.title('Household Size Distribution in Percentages')
            plt.xlabel('Size')
            plt.ylabel('Percentage (%)')
            plt.xticks(range(1, len(distribution_percentages) + 1))
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}%'))
            plt.grid(axis='y')
            st.pyplot(plt)

        # Plot: Parental Status Distribution
        with col4:
            # Assuming df has 'parental_status' column, if not replace with correct column
            parental_pct = df['parental_status'].value_counts(normalize=True) * 100
            # parental_pct = parental_pct.sort_index()  # Sort by index to maintain original order

            plt.figure(figsize=(5, 4))
            parental_pct.plot(kind='bar', color='skyblue', edgecolor='black')
            plt.title('Parental Status Distribution in Percentages')
            plt.xlabel('Parental Status')
            plt.ylabel('Percentage (%)')
            plt.xticks(rotation=0)  # Rotate labels to make them readable
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}%'))
            plt.grid(axis='y')
            st.pyplot(plt)

    else:
        st.write("Please run the eligibility calculation first.")