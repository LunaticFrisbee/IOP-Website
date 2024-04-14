import React from "react";
import { Table } from "antd";

import styles from "./Forecast.module.css";

const Forecast = (props) => {
  const columns = [
    {
      title: "Prediction",
      dataIndex: "Prediction",
      key: "Prediction",
      align: "center",
    },
    {
      title: "Forecast",
      dataIndex: "Forecast",
      key: "Forecast",
      align: "center",
    },
  ];
  return (
    <div className={styles.mainContainer}>
      <div className={styles.table}>
        <Table
          dataSource={props.forecast}
          columns={columns}
          bordered={true}
          pagination={{ pageSize: 24 }}
        />
      </div>
      <img
        alt="Plot for forecast and testLoad"
        src={"data:image/png;base64," + props.plot}
      ></img>
    </div>
  );
};

export default Forecast;
