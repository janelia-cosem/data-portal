import React, { useState } from "react";
import PropTypes from "prop-types";
import {checkWebGL2} from "../api/util"
import { Dataset} from "../api/datasets";

export interface ContextProps {
  neuroglancerAddress: string,
  dataBucket: string,
  webGL2Enabled: boolean,
  datasetsLoading: boolean,
  datasets: Map<string, Dataset>
}

interface AppContext {
  appState: ContextProps
  setAppstate: () => null
}

const contextDefault: ContextProps = {
  neuroglancerAddress: "http://neuroglancer-demo.appspot.com/#!",
  dataBucket: 'janelia-cosem-datasets',
  webGL2Enabled: checkWebGL2(),
  datasetsLoading: false,
  datasets: new Map()
}

export const AppContext = React.createContext<AppContext>({
  appState: contextDefault,
  setAppstate: () => null
});

export const AppProvider = (props: any) => {

  const [state, setState] = useState<ContextProps>(contextDefault);
  const { children } = props;
  return (
    <AppContext.Provider value={[state, setState]}>
      {children}
    </AppContext.Provider>
  );
}

AppProvider.propTypes = {
  children: PropTypes.object.isRequired
}
