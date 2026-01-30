import numpy as np
import pandas as pd

def generate_real_estate_data(n_samples: int = 1000, random_state: int = 42):
    np.random.seed(random_state)

    # Features
    square_feet = np.random.randint(500, 3500, n_samples)               # house size
    num_bedrooms = np.random.randint(1, 6, n_samples)                   # 1-5
    num_bathrooms = np.random.randint(1, 4, n_samples)                  # 1-3
    dist_to_city_center = np.round(np.random.uniform(1, 30, n_samples), 2)  # km
    age_of_house = np.random.randint(0, 51, n_samples)                  # years

    # Price formula (synthetic but semi-realistic)
    price = (
        square_feet * 300
        + num_bedrooms * 50000
        + num_bathrooms * 30000
        - dist_to_city_center * 2000
        - age_of_house * 1000
        + np.random.normal(0, 25000, n_samples)  # noise
    )

    # Build DataFrame
    df = pd.DataFrame({
        "square_feet": square_feet,
        "num_bedrooms": num_bedrooms,
        "num_bathrooms": num_bathrooms,
        "dist_to_city_center": dist_to_city_center,
        "age_of_house": age_of_house,
        "price": price.astype(int)
    })

    return df

if __name__ == "__main__":
    df = generate_real_estate_data()
    df.to_csv("corpus/raw_prices.csv", index=False)
    print("✅ Dataset generated and saved as data/dataset.csv")
