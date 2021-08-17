<template>
  <div class="line">
    <div class="title">Message Category Distribution</div>
    <div id="line-chart"></div>
  </div>
</template>

<script>
import { Column } from "@antv/g2plot";

export default {
  props: {
    lineData: Array,
  },
  watch: {
    lineData: (lineData) => {
      if (lineData.length > 0) {
        console.log(lineData.length)
        const columnPlot = new Column("line-chart", {
          data: lineData,
          xField: "name",
          yField: "value",
          label: {
            position: "middle", 
            style: {
              fill: "#FFFFFF",
              opacity: 0.6,
            },
          },
          xAxis: {
            label: {
              autoHide: false,
              autoRotate: true,
            },
          },
        });
        columnPlot.render();
      }
    },
  },
  mounted() {},
  methods() {},
};
</script>

<style scoped>
.line {
  width: calc(100vw - 40px);
  box-sizing: border-box;
  box-shadow: #435a6f 0px 0px 1px 0px;
  padding: 16px 24px 24px;
}
.title {
  font-weight: 500;
  text-align: left;
}
#line-chart {
  height: 490px;
  width: 100%;
}
</style>
