

$( document ).ready(function() {

    mapVis = new MapboxGL("map", 12.2, 0, false, "data/CrimeDone.json");
    mapVis2 = new MapboxGL("map2", 13, 45, true ,"data/CrimeDone.json");


    var dummyData = GetDummyDataForParallelCoordinate();
    var ParallelCoordinatesDiv = document.getElementById("ParallelCoordinatesDiv");

    new ParallelCoordinatesInit("ParallelCoordinatesDiv", dummyData, "design space - Parallel coordinate" );

});