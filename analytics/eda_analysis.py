from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

MODULE_DIR = Path(__file__).parent


class EdaAnalysis:
    # Profiling
    def write_sns_titanic_to_csv(self):
        titanic_lit = "titanic"
        # Same path clean_data_set() reads, so this check actually recognizes
        # the committed offline fallback instead of looking at the CWD.
        file_path = MODULE_DIR / f"{titanic_lit}.csv"
        # Check existence using the .exists() method
        if file_path.exists():
            print(f"⚠️ '{file_path.name}' already exists.")
            titanic = pd.read_csv(file_path)
        else:
            titanic = sns.load_dataset(f'{titanic_lit}')
            titanic.to_csv(file_path, index=False)

        print(f"Original shape: {titanic.shape}")
        

    # Cleaning
    def clean_data_set(self):
        
        # 1. Load data from the CSV file
        df = pd.read_csv(MODULE_DIR / 'titanic.csv')

        # 2. Measure the exact initial missing value percentages across the original data
        missing_pct = df.isnull().mean() * 100

        print("--- Initial Missingness Measurement ---")
        for col in df.columns:
            if missing_pct[col] > 0:
                print(f"Column: {col:<12} | Exact Missing Rate: {missing_pct[col]:.2f}%")
        print("-" * 40)

        # 3. Apply your threshold processing pipeline
        for col in list(df.columns):  # Safely iterate using a static list of original columns
            if col not in df.columns:
                continue
                
            pct = missing_pct[col]
            if pct == 0:
                continue
                
            # Rule 1: Under 5% missing -> Drop rows
            elif pct < 5:
                print(f"Action: Dropped rows in [{col}] (Measured: {pct:.2f}%)")
                df = df.dropna(subset=[col])
                
            # Rule 2: 5% to 30% missing -> Impute
            elif 5 <= pct <= 30:
                print(f"Action: Imputed values in [{col}] (Measured: {pct:.2f}%)")
                if df[col].dtype.kind in 'iufc':
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])
                    
            # Rule 3: Over 30% missing -> Explicitly Drop Column
            elif pct > 30:
                print(f"⚠️ Action: Dropping entire column [{col}] (Measured: {pct:.2f}%)")
                df = df.drop(columns=[col])

        print("\n--- Final Status ---")
        print(f"Columns after processing: {df.shape[1]}")
        print(f"Remaining columns: {list(df.columns)}")
        
        return df


    # 3. IQR Outlier Function
    def _calculate_outliers(self, series):
        clean_series = series.dropna()
        q1 = clean_series.quantile(0.25)
        q3 = clean_series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
            
        outliers = clean_series[(clean_series < lower_bound) | (clean_series > upper_bound)]
        return outliers.count(), lower_bound, upper_bound


    # Data story - Univariate analysis
    def univariate_analysis(self,df):
        # 2. Set up the plotting grid (2 rows, 2 columns)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Univariate Analysis: Age & Fare Distributions', fontsize=16, fontweight='bold')

        # --- AGE PLOTS ---
        sns.histplot(df['age'].dropna(), kde=True, ax=axes[0, 0], color='skyblue')
        axes[0, 0].set_title('Age Distribution (Histogram)')

        sns.boxplot(x=df['age'], ax=axes[0, 1], color='skyblue')
        axes[0, 1].set_title('Age Range (Box Plot)')

        # --- FARE PLOTS ---
        sns.histplot(df['fare'], kde=True, ax=axes[1, 0], color='salmon')
        axes[1, 0].set_title('Fare Distribution (Histogram)')

        sns.boxplot(x=df['fare'], ax=axes[1, 1], color='salmon')
        axes[1, 1].set_title('Fare Range (Box Plot)')

        plt.tight_layout()
        plt.show()

        # Calculate for both columns
        age_outliers, age_lb, age_ub = self._calculate_outliers(df['age'])
        fare_outliers, fare_lb, fare_ub = self._calculate_outliers(df['fare'])

        # 4. Central Tendencies for Fare
        fare_mean = df['fare'].mean()
        fare_median = df['fare'].median()
        fare_mode = df['fare'].mode()[0]

        # Print Text Summary
        print(f"--- Outlier Report (IQR Rule) ---")
        print(f"Age Outliers: {age_outliers} points (Outside range {age_lb:.2f} to {age_ub:.2f})")
        print(f"Fare Outliers: {fare_outliers} points (Outside range {fare_lb:.2f} to {fare_ub:.2f})\n")
        print(f"--- Fare Central Tendency Metrics ---")
        print(f"Mean:   ${fare_mean:.2f}")
        print(f"Median: ${fare_median:.2f}")
        print(f"Mode:   ${fare_mode:.2f}")
        

    # Data story - Bivariate analysis

    def bivariate_analysis(self, df):
        
        print(" ================ Bi Variate Analysis ==================== \n")
        
        # (a) Survival rate broken down by Sex
        print("(a) Survival Rate by Sex:")
        for s in df['sex'].unique():
            mask = df['sex'] == s
            rate = df.loc[mask, 'survived'].mean()
            print(f"    {s.capitalize()}: {rate * 100:.2f}%")

        # (b) Survival rate broken down by Pclass
        print("\n(b) Survival Rate by Pclass:")
        for p in sorted(df['pclass'].unique()):
            mask = df['pclass'] == p
            rate = df.loc[mask, 'survived'].mean()
            print(f"    Class {p}: {rate * 100:.2f}%")

        # (c) Survival rate broken down by Sex and Pclass together
        print("\n(c) Survival Rate by Sex AND Pclass Together:")
        for s in df['sex'].unique():
            for p in sorted(df['pclass'].unique()):
                # Boolean combination using the bitwise AND (&) operator
                mask = (df['sex'] == s) & (df['pclass'] == p)
                rate = df.loc[mask, 'survived'].mean()
                print(f"    {s.capitalize()} in Class {p}: {rate * 100:.2f}%")
                
        # 2. Compute 6x6 correlation matrix (Restricted & flags excluded)
        numeric_cols = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']
        corr_matrix = df[numeric_cols].corr()

        # 3. Render Heatmap Visual
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            cmap='coolwarm', 
            fmt=".3f", 
            vmin=-1, 
            vmax=1, 
            linewidths=0.5
        )
        plt.title('Correlation Heatmap: Core Titanic Numerical Features', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Get the top 2 strongest unique off-diagonal correlations
        top_2 = corr_matrix.abs().unstack().drop_duplicates().sort_values(ascending=False).iloc[1:3]

        print("Absolute Correlation for Top 2 :\n",top_2)
        
    
    def multivairate_analysis(self, df):
        # ==========================================
        # CHART 1: Survival Rate by Sex (Bar Chart)
        # ==========================================
        plt.figure()
        sns.barplot(data=df, x='sex', y='survived', errorbar=None, palette='muted')
        plt.title('Survival Rate by Biological Sex', fontsize=14, fontweight='bold')
        plt.xlabel('Sex')
        plt.ylabel('Survival Rate (Probability)')
        plt.tight_layout()
        plt.savefig(MODULE_DIR / '1_survival_by_sex.png')
        plt.show()


        # ==========================================
        # CHART 2: Class & Gender Interaction (Heatmap)
        # ==========================================
        plt.figure()
        # Calculate the mean survival rate for every combination of Class and Sex
        pivot_df = df.pivot_table(values='survived', index='sex', columns='pclass', aggfunc='mean')

        sns.heatmap(pivot_df, annot=True, cmap='RdYlGn', fmt=".2f", vmin=0, vmax=1, 
                    cbar_kws={'label': 'Survival Probability'})
        plt.title('Survival Rate by Class and Sex', fontsize=14, fontweight='bold')
        plt.xlabel('Passenger Class (Pclass)')
        plt.ylabel('Sex')
        plt.tight_layout()
        plt.savefig(MODULE_DIR / '2_survival_heatmap.png')
        plt.show()


        # ==========================================
        # CHART 3: Age Distribution (Box Plot)
        # ==========================================

        # Map numeric survival to human-readable strings for labels
        df['survival_label'] = df['survived'].map({0: 'Perished', 1: 'Survived'})

        plt.figure()
        sns.boxplot(data=df, x='survival_label', y='age', palette='Set2')
        plt.title('Age Distribution by Survival Status', fontsize=14, fontweight='bold')
        plt.xlabel('Survival Status')
        plt.ylabel('Age (Years)')
        plt.tight_layout()
        plt.savefig(MODULE_DIR / '3_age_distribution_box.png')
        plt.show()

        # ==========================================
        # CHART 4: Wealth vs. Age Impact (Scatter Plot)
        # ==========================================
        plt.figure()
        # We use a log scale because a few passengers paid exceptionally high outlier fares (e.g., £512)
        sns.scatterplot(data=df, x='age', y='fare', hue='survival_label', 
                        style='survival_label', palette={'Perished': '#e74c3c', 'Survived': '#2ecc71'}, 
                        alpha=0.8, s=60)
        plt.yscale('symlog')  # Handles extreme fare discrepancies smoothly
        plt.title('Impact of Wealth (Fare) and Age on Survival', fontsize=14, fontweight='bold')
        plt.xlabel('Age (Years)')
        plt.ylabel('Fare Paid (Log Scale)')
        plt.legend(title='Status')
        plt.tight_layout()
        plt.savefig(MODULE_DIR / '4_fare_vs_age_scatter.png')
        plt.show()
        
        


    # Data story - Exploratroy check
    def exploratory_check(self, df):
        # Before statistics 
        before_stats = df[['age', 'fare']].agg(['mean', 'std']).T

        # Apply the Z-score formula manually as an EDA check
        # Note: For age, missing values (NaNs) are preserved natively by pandas during vector math
        df['age_z'] = (df['age'] - df['age'].mean()) / df['age'].std()
        df['fare_z'] = (df['fare'] - df['fare'].mean()) / df['fare'].std()


        # Before statistics 
        after_stats = df[['age_z', 'fare_z']].agg(['mean', 'std']).T


        # 5. Display a clean before/after summary comparison
        print("=== BEFORE STANDARDIZATION ===")
        print(before_stats.rename(columns={'mean': 'Original Mean', 'std': 'Original Std'}).round(3))

        print("\n=== AFTER STANDARDIZATION ===")
        print(after_stats.rename(columns={'mean': 'Transformed Mean', 'std': 'Transformed Std'}).round(3))
        
        # 6. Plot overlaid distributions to visually verify the scale shift
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Age Overlay Plot
        sns.kdeplot(df['age'], label='Original Age', shade=True, ax=axes[0], color='skyblue')
        sns.kdeplot(df['age_z'], label='Z-scaled Age', shade=True, ax=axes[0], color='blue')
        axes[0].set_title('Age Scale Shift Comparison')
        axes[0].set_xlabel('Value')
        axes[0].legend()

        # Fare Overlay Plot
        sns.kdeplot(df['fare'], label='Original Fare', shade=True, ax=axes[1], color='salmon')
        sns.kdeplot(df['fare_z'], label='Z-scaled Fare', shade=True, ax=axes[1], color='red')
        axes[1].set_title('Fare Scale Shift Comparison')
        axes[1].set_xlabel('Value')
        axes[1].legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    eda = EdaAnalysis()
    #Profiling
    eda.write_sns_titanic_to_csv()
    # Cleaning
    df = eda.clean_data_set()
    #Univariate Analysis
    eda.univariate_analysis(df)
    #Bivariate Analysis
    eda.bivariate_analysis(df)
    # Multivariate "data story"
    eda.multivairate_analysis(df)
    # Exploratory check 
    eda.exploratory_check(df)
    