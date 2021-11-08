var needle = require('needle');
var fs = require('fs');

const ENDPOINT = 'https://api.asrank.caida.org/v2/restful/asns/';
let CURRENT_PAGE = 0;
let all = [];
let has_next = true;


async function fetchData() {
    while(has_next) {
        let url = ENDPOINT;
        if (CURRENT_PAGE > 0) {
            // pass the items that I have already pulled
            url += "?offset=" + (500 * CURRENT_PAGE);
        }
        console.log(url, CURRENT_PAGE);
        // Make the request and wait await to finish
        const res = await needle('get', url);
        const data = res.body.data.asns;

        //pass data in a Global Array
        data.edges.forEach(edge => {
            all.push({
                asn: edge.node.asn,
                rank: edge.node.rank,
                source: edge.node.source,
                longitude: edge.node.longitude,
                latitude: edge.node.latitude,
                numberAsns: edge.node.cone.numberAsns,
                numberPrefixes: edge.node.cone.numberPrefixes,
                numberAddresses: edge.node.cone.numberAddresses,
                iso: edge.node.country.iso,
                total: edge.node.asnDegree.total,
                customer: edge.node.asnDegree.customer,
                peer: edge.node.asnDegree.peer,
                provider: edge.node.asnDegree.provider,
            })
        })
        // check for the existence of next page 
        has_next = data.pageInfo && data.pageInfo.hasNextPage;
        CURRENT_PAGE++;
    }
}
/*
First I call the fetchData function which brings me all the data.
Then I store the data in a string in order to pass them to a json file
*/
(async () => {
    await fetchData();
    var json = JSON.stringify(all);
    function tmpfn() {}
    fs.writeFile('asns.json', json, 'utf8', tmpfn);
})();
