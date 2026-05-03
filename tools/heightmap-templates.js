"use strict";

const heightmapTemplates = (function () {

  const lowIsland = `Hill 1 90-99 20-22 20-22
    Hill 1 90-99 20-22 77-80
    Hill 1 90-99 77-80 20-22
    Hill 1 90-99 77-80 77-80
    Smooth 2 0 0 0
    Hill 10 30-35 20-80 20-80
    Range 1 40-50 45-55 45-55
    Trough 2-3 20-30 15-85 20-30
    Trough 2-3 20-30 15-85 70-80
    Hill 1.5 10-15 5-15 20-80
    Hill 1 10-15 85-95 70-80
    Pit 4 90-95 46-54 46-54
    Pit 5-7 15-25 15-85 20-80
    Multiply 0.4 20-100 0 0
    Mask 4 0 0 0`;

  const highIsland = `Hill 1 97-100 20-22 20-22
    Hill 1 97-100 20-22 77-80
    Hill 1 97-100 77-80 20-22
    Hill 1 97-100 77-80 77-80
    Add 7 all 0 0
    Hill 5-6 20-30 25-55 45-55
    Range 1 40-50 45-55 45-55
    Multiply 0.8 land 0 0
    Mask 3 0 0 0
    Smooth 2 0 0 0
    Trough 2-3 20-30 20-30 20-30
    Trough 2-3 20-30 60-80 70-80
    Hill 1 10-15 60-60 50-50
    Hill 1.5 13-16 15-20 20-75
    Pit 4 90-95 46-54 46-54
    Range 1.5 30-40 15-85 30-40
    Range 1.5 30-40 15-85 60-70
    Pit 3-5 10-30 15-85 20-80`;

  return {
    highIsland: {id: 1, name: "Hight Island", template: highIsland, probability: 10},
    // lowIsland: {id: 2, name: "Low Island", template: lowIsland, probability: 10},
  };
})();
