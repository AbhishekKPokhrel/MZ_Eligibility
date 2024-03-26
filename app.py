import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import streamlit as st


### User inputs #####

# Function to calculate the percentage eligible
def calculate_percentage_eligible(num_employees, hourly_pct, hr_min, hr_median, hr_max, avg_dep, state):
    ### General assumptions ####

    avg_weekly_hrs = 40 # average hours worked per week

    weeks_in_yrs = 52 # weeks employees are expected to work each year



    # User-defined distributions for household size and parental status
    distribution_percentages = {
     1: 0.285, 2: 0.35, 3: 0.15, 4: 0.125, 5: 0.06, 6: 0.03
    }

    percentages = {
        "Single": 0.54,   # employee without partner or child
        "Dual": 0.115,    # employee with partner and no child
        "Single_P": 0.115, # employee with child and no partner
        "Dual_P": 0.23     # employee with both partner and child
    }




    ############################
    #### Generating Salaries ####
    ############################

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
    df_salary = pd.DataFrame(data)


    ## Check ##
    below_thres = np.sum(df_salary['hourly_salary'] < hr_median)
    #print(below_thres)


    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df_salary['hourly_salary'], bins=20, color='skyblue', edgecolor='black')  # Adjust bins as needed
    plt.title('Hourly Pay Distribution')
    plt.xlabel('Salary')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    #fig, ax = plt.subplots()
    #ax.hist(df_salary['hourly_salary'], bins=20, color='skyblue', edgecolor='black')
    #st.pyplot(fig)

    df = df_salary

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

    # Function to calculate total household salary based on parental status
    def calculate_total_household_salary(row):
        if row['parental_status'] in ['Dual', 'Dual_P']:
            return row['annual_salary'] * 2
        else:
            return row['annual_salary']
        
        
        
    # Apply the function to create the 'total_household_salary' column
    df['total_household_income'] = df.apply(calculate_total_household_salary, axis=1)

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


    
    # Use the input variables in the model's logic to calculate the final outcome
    # Here, just return a dummy percentage for demonstration
    return eligible_count, percentage_eligible  # Placeholder percentage



# Streamlit app layout
# Using markdown for title with adjusted size
#st.markdown("## ChildCare Subsidy Eligibility Calculator")


st.title("ChildCare Subsidy Eligibility")

# Organize inputs side by side using columns
# First row of inputs
col1, col2, col3 = st.columns(3)
with col1:
    num_employees = st.number_input("Number of Employees", min_value=1, value=1000)
with col2:
    avg_dep = st.number_input("Average Dependents", value=3)
with col3:
    state = st.text_input("State", value='CA')

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

# Calculate button
if st.button("Calculate Eligibility"):
    eligible_count, percentage_eligible = calculate_percentage_eligible(
        num_employees, hourly_pct, hr_min, hr_median, hr_max, avg_dep, state
    )
    # Enhanced display using st.metric
    #st.metric(label="Total Employees Eligible for ChildCare Subsidies", value=f"{eligible_count:.2f}%", delta=None)
    
    # Use Markdown with custom CSS for double underlining
    st.markdown(f"""
        <div style='text-align: center; color: black;'>
            <span style='font-size: x-large; font-weight: bold;'>
                <span style='border-bottom: 3px double; padding-bottom: 2px;'>
                    {eligible_count} ({percentage_eligible:.1f}%)
                </span> Employees Eligible
            </span>
        </div>
        """, unsafe_allow_html=True)
