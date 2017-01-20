

function MapboxGL(theDiv,zoomFactor, _pitch, navigable, data){
    mapboxgl.accessToken = 'YOUR TOKEN';
    this.map = new mapboxgl.Map({
                    container: theDiv, //ex 'map', // container id
                    style: 'mapbox://styles/designju/cild5molw008w9pkne3c8ydgw', //hosted style id
                    center: [-71.067273, 42.314914 ], // starting position
                    zoom: zoomFactor, //2, // starting zoom
                    pitch: _pitch, // pitch in degrees 45
                    bearing: 0.0 // bearing in degrees
                    });
    this.dataPath = data;
    this.theDiv = theDiv;

    if(true){
        this.map.addControl(new mapboxgl.Navigation());
        this.map.on("viewreset", this.Update);
        this.map.on("move", this.Update);
    }
    this.LoadData();
};
MapboxGL.prototype.LoadData=function() {
    var vis = this;
    queue().defer(d3.csv, vis.dataPath)
            .await(function(error, data){
                vis.rawData = data
                vis.DataPrcess();
            });
    $( window ).resize(function() {
            vis.Update();
    });
};
MapboxGL.prototype.DataPrcess = function() {
    var vis = this;
    // after data process, we need to update the Visdata to visualze
    vis.VisData = this.rawData  
	vis.VisMap();
};
MapboxGL.prototype.VisMap = function(){
    var vis = this;
    this.theSvg = d3.select("#" + this.theDiv)
                        .append("svg")
                        .attr("class", "d3Canvas")
                        .attr("id", "d3Canvas")
                        .attr("height", "100%")
                        .attr("width", "100%")
                        .attr("z-index", 3)
                        .style("position","absolute");

    this.circle = this.theSvg
                        .selectAll("circle")
                        .data(vis.VisData)
                        .enter()
                        .append("circle")
                        .attr({
                            "class":"cir",
                            // "stroke": "red",
                            "fill": "purple",
                            "fill-opacity": 0.2,
                            "r":2
                        })
                        .on("click", function(d){
                            var thePos = [+d.pos.split(",")[0] , +d.pos.split(",")[1]]
                            $("#maplogCrime").html("<p>Site_EUI:"+d.Site_EUI +"<br>" +
                                              "lat:" + (+thePos[1]) +"<br>" +
                                              "long:"+ (+thePos[0]) +"</p>");

                        });
    this.Update()           
};

MapboxGL.prototype.Update = function() {
    var vis = this;
    vis.circle
        .attr("cx", function(d){
            // console.log(d)
            // console.log(typeof(d))
            // console.log(d.pos.split(",")[0]);
            var thePos = [+d.pos.split(",")[0] , +d.pos.split(",")[1]]
            return vis.Projection(thePos).x;
        })
        .attr("cy", function(d){
            var thePos = [+d.pos.split(",")[0] , +d.pos.split(",")[1]]
            return vis.Projection(thePos).y; 
        })
        .attr("r", function(d){
            var value = (+d.Site_EUI) * 0.00001
            if(value == 0){
                return 4; 
            }else{
                return 4 + value + value;
            }
        });
}
MapboxGL.prototype.Projection = function(d) {
    return this.map.project(new mapboxgl.LngLat(+d[0], +d[1]));
};



    
