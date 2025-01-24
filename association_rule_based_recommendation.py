#########################
# Association Rule Based Recommendation
#########################

# Armut, Turkey's largest online service platform, connects service providers with customers. With just a few clicks
# on a computer or smartphone, users can easily access services such as cleaning, renovation, and moving. Using a
# dataset containing the services and categories purchased by users, an Association Rule Learning-based
# recommendation system is to be created.

# The dataset consists of the services customers have purchased and their respective categories.
# It also includes the date and time of each service purchase.

# UserId: Customer ID

# ServiceId: Anonymized services for each category. (Example: Under the cleaning category, a sofa cleaning service)
# A ServiceId can exist under different categories and represent different services under each category.
# (Example: A service with CategoryId 7 and ServiceId 4 may represent radiator cleaning,
# whereas the same ServiceId under CategoryId 2 may represent furniture assembly.)

# CategoryId: Anonymized categories. (Example: Cleaning, moving, renovation)

# CreateDate: The date the service was purchased

import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
from mlxtend.frequent_patterns import apriori, association_rules

#########################
# TASK 1: Data Preparation
#########################

# Step 1: Read the armut_data.csv file.
df_ = pd.read_csv("Recommendation Systems/datasets/armut_data.csv")
df = df_.copy()
df.head()

# Step 2: ServiceId represents a different service for each CategoryId.
# Combine ServiceId and CategoryId with "_" to create a new variable representing the services.
df["Hizmet"] = df.apply(lambda row: f"{row['ServiceId']}_{row['CategoryId']}", axis=1)
df.head()

# Step 3: The dataset includes the date and time of the services, but there is no basket (invoice) definition. To
# apply Association Rule Learning, we need to create a basket definition. Here, a basket is defined as the monthly
# services purchased by each user. For example: User ID 7256, in August 2017, purchased services 9_4 and 46_4 as one
# basket; in October 2017, services 9_4 and 38_4 as another basket. Baskets need to be uniquely identified. To
# achieve this, first create a new date variable containing only the year and month. Combine UserID with the new date
# variable using "_" to assign it to a new variable called ID.

df["CreateDate"] = pd.to_datetime(df["CreateDate"])
df.head()

df["New_Date"] = df["CreateDate"].dt.strftime("%Y-%m")
df.head()

df["SepetID"] = df.apply(lambda row: f"{row['UserId']}_{row['New_Date']}", axis=1)
df.head()

#########################
# TASK 2: Generate Association Rules
#########################

# Step 1: Create a pivot table for the basket services as shown below.

# Service         0_8  10_9  11_11  12_7  13_11  14_7  15_1  16_8  17_5  18_4..
# BasketID
# 0_2017-08        0     0      0     0      0     0     0     0     0     0..
# 0_2017-09        0     0      0     0      0     0     0     0     0     0..
# 0_2018-01        0     0      0     0      0     0     0     0     0     0..
# 0_2018-04        0     0      0     0      0     1     0     0     0     0..
# 10000_2017-08    0     0      0     0      0     0     0     0     0     0..

invoice_product_df = (df.groupby(["SepetID", "Hizmet"])["Hizmet"].count().unstack().fillna(0).
                      applymap(lambda x: 1 if x > 0 else 0))

invoice_product_df.head()

# Step 2: Generate association rules.

frequent_itemsets = apriori(invoice_product_df, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.head()


# Step 3: Use the arl_recommender function to recommend services for a user
# who most recently purchased the service "2_0".

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    # Sort rules by lift in descending order (to find the most relevant products).
    # Alternatively, rules can be sorted by confidence, depending on preference.
    recommendation_list = []  # Create an empty list for recommended products.
    # antecedents: X
    # The items are returned as frozenset (immutable sets). Combine index and service.
    for i, product in sorted_rules["antecedents"].items():
        for j in list(product):  # Iterate through products in antecedents (X):
            if j == product_id:  # If the requested product is found:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))
                # Add the consequents (Y) of this rule to the recommendation list.

    # To avoid duplicates in the recommendation list:
    # For example, in 2-item or 3-item combinations, the same product might appear multiple times.
    # Use the unique property of dictionaries to eliminate duplicates.
    recommendation_list = list({item for item_list in recommendation_list for item in item_list})
    return recommendation_list[:rec_count]  # Return the desired number of recommended products.


arl_recommender(rules, "2_0", 4)
