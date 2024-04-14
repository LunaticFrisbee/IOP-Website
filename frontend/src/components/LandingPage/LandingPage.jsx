import React, { useState } from "react";
import axios from "axios";

import styles from "./LandingPage.module.css";
import Forecast from "../Forecast/Forecast";

const LandingPage = () => {
	const [prediction, setPrediction] = useState(0);
	const [plot, setPlot] = useState("");
	const [forecast, setForecast] = useState("");

	const uploadHandler = (e) => {
		if (prediction !== 0 && e.target.files && e.target.files[0]) {
			var bodyFormData = new FormData();
			bodyFormData.append("dataSet", e.target.files[0]);
			axios({
				method: "post",
				url:
					process.env.REACT_APP_URLPATH +
					"prediction/holtwinter?npreds=" +
					prediction,
				data: bodyFormData,
			})
				.then(function (response) {
					setPlot(JSON.parse(response.data.plot));
					var data = JSON.parse(response.data.forecast);
					var cnt = 0;
					var cast = Object.keys(data).map((key) => {
						cnt = cnt + 1;
						return { Prediction: cnt, Forecast: data[key] };
					});
					setForecast(cast);
				})
				.catch((error) => console.log(error));
		}
	};

	const predictionHandler = (e) => {
		setPrediction(e.target.value * 24);
	};

	return (
		<>
			{!plot && (
				<div className={styles.mainContainer}>
					<div className={styles.content}>
						<h1>Load Forecasting</h1>
						<p>
							Discover the ultimate resource for accurate electrical load
							forecasts and their dynamic fluctuations
						</p>
						<label htmlFor="prediction" className={styles.predictionLabel}>
							Predictions:{" "}
						</label>
						<input
							id="prediction"
							className={styles.predictionInput}
							type="number"
							onChange={predictionHandler}
							title="predictions"
							placeholder="Number of Days:"
						/>
						<input type="file" onChange={uploadHandler} />
					</div>
				</div>
			)}
			{plot && <Forecast forecast={forecast} plot={plot} />}
		</>
	);
};

export default LandingPage;
