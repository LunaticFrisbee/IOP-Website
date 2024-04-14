import React from "react";
import { NavLink } from "react-router-dom";

import styles from "./Navbar.module.css";

const Navbar = () => {
  return (
    <nav className={styles.navBar}>
      <div className={styles.navLogo}>Electrical Load Forecasting</div>
      <NavLink to="/home">Home</NavLink>
    </nav>
  );
};

export default Navbar;
