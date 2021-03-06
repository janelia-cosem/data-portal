export interface queryResponse {
  isLoading: boolean;
  isError: boolean;
  data: any;
  error: any;
}

export function fetchAnalysisResults(cypher: string) {
  const options = {
    method: "POST",
    headers: {
      accept: "application/json",
      "Content-Type": "application/json",
      Authorization: `Basic bmVvNGo6QWlFc2h0ZUMwUzNNIQ==`
    },
    body: JSON.stringify({ statements: [{ statement: cypher }] })
  };

  return fetch("https://cosem-db.janelia.org:7473/db/graph.db/tx", options)
    .then(response => response.json())
    .then(res => {
      if (res.results) {
        return res.results[0];
      }
      return;
    });
}


