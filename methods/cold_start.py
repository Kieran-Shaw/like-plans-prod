import logging

import Levenshtein
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ColdStart:
    def __init__(self, data: dict, logger: logging.Logger) -> None:
        self.census_plan_admin_name = data["census_plan_admin_name"]
        self.census_carrier_name = data["census_carrier_name"]
        self.available_plans = data["available_plans"]
        self.vectorizer = TfidfVectorizer()

        # initialize the logger
        self.logger = logger
        self.logger.info(
            "Initialized cold start",
            extra={
                "client_name": self.census_carrier_name,
                "client_id": self.census_plan_admin_name,
            },
        )

    def predict_plans(self):
        available_plans_df = self._available_plans_to_df()
        most_similar_carrier_name = self._best_match_carrier(df=available_plans_df)
        filtered_df = self._filter_dataframe_for_carrier(
            df=available_plans_df, similar_carrier_name=most_similar_carrier_name
        )
        top_three_similar = self._return_top_three_similar(filtered_df=filtered_df)

        return top_three_similar

    def _available_plans_to_df(self):
        df = pd.DataFrame(self.available_plans)
        self.logger.info(
            "Compelted available plans to dataframe", extra={"length": len(df)}
        )
        return df

    def _best_match_carrier(self, df: pd.DataFrame):
        # list of unique carriers in df
        unique_carrier_names = list(df["carrier"].unique())
        # Calculate Levenshtein distance to known carrier names
        distances = [
            Levenshtein.distance(self.census_carrier_name, carrier)
            for carrier in unique_carrier_names
        ]

        # Find the most similar carrier name based on minimum distance
        most_similar_index = np.argmin(distances)
        most_similar_carrier_name = unique_carrier_names[most_similar_index]
        print(most_similar_carrier_name)

        # log out the most similar
        self.logger.info(
            "Most similar carrier name complete",
            extra={
                "census_carrier)name": self.census_carrier_name,
                "most_similar_carrier_name": most_similar_carrier_name,
            },
        )

        return most_similar_carrier_name

    def _filter_dataframe_for_carrier(
        self, df: pd.DataFrame, similar_carrier_name: str
    ):
        # copy dataframe
        new_df = df.copy()

        # filter new dataframe
        filtered_df = new_df[new_df["carrier"] == similar_carrier_name]
        filtered_df.reset_index(drop=True, inplace=True)

        # log out the filtered dataframe
        self.logger.info(
            "Dataframe filtered for most similar carrier",
            extra={"length": len(filtered_df)},
        )

        return filtered_df

    def _return_top_three_similar(self, filtered_df: pd.DataFrame):
        # Vectorize the filtered_df names
        name_vectors = self.vectorizer.fit_transform(filtered_df["name"])

        # Find the most similar plan
        query_vector = self.vectorizer.transform([self.census_plan_admin_name])
        similarities = cosine_similarity(query_vector, name_vectors)

        # Get the indices of the top three most similar plans
        top_three_indices = similarities.argsort()[0][-3:][::-1]

        # Get the most similar plan names and IDs
        top_three_names = filtered_df.loc[top_three_indices, "name"].values
        top_three_ids = filtered_df.loc[top_three_indices, "id"].values
        top_three_scores = similarities[0][top_three_indices]

        # Package the information into a dictionary
        response_object = {
            "1": {
                "name": top_three_names[0],
                "id": top_three_ids[0],
                "score": top_three_scores[0],
            },
            "2": {
                "name": top_three_names[1],
                "id": top_three_ids[1],
                "score": top_three_scores[1],
            },
            "3": {
                "name": top_three_names[2],
                "id": top_three_ids[2],
                "score": top_three_scores[2],
            },
        }

        # log out the response object
        self.logger.info(
            "Most similar plans returned",
            extra={"response_object": response_object},
        )

        return response_object
