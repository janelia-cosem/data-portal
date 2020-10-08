import React from "react";
import Typography from "@material-ui/core/Typography";
import { makeStyles } from "@material-ui/core/styles";

import cosemLogo from "../COSEM_logo_invert_transparent.png";
import fibsemLogo from "../fibsem_logo.png";

const cosemURL = "https://www.janelia.org/project-team/cosem"
const hessURL = "https://www.janelia.org/lab/hess-lab"
const fibsemURL = "https://www.janelia.org/lab/fib-sem-technology"

const useStyles = makeStyles(theme => ({
  link: {
    color: theme.palette.primary.contrastText,
    textDecoration: "none",
    marginLeft: "1em",
    display: "inline-block"
  },
  cosemLogo: {
    maxHeight: "70px",
    marginRight: theme.spacing(2)
  },
  footer: {
    padding: '1em 0',
    background: '#333',
    color: '#fff',
    textAlign: 'center'
  }
}));


export default function Footer() {
  const classes = useStyles();
  return (
    <footer className={classes.footer}>
      <a
        href={cosemURL}
        className={classes.link}
        target="_blank"
        rel="noopener noreferrer"
      >
        <img src={cosemLogo} alt="COSEM" className={classes.cosemLogo} />
      </a>
      <a href={fibsemURL} target="_blank" rel="noopener noreferrer">
        <img
          src={fibsemLogo}
          alt="FIB-SEM Technology"
          className={classes.cosemLogo}
        />
      </a>
      <a
        href={hessURL}
        className={classes.link}
        target="_blank"
        rel="noopener noreferrer"
      >
        <Typography variant="h6" color="inherit" noWrap>
          Hess Lab
        </Typography>
      </a>
      <p>&copy; 2020 HHMI</p>
    </footer>
  );
}