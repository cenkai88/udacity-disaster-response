<template>
  <div class="pie">
    <div class="title">Message Genre Distribution</div>
    <div id="chart"></div>
  </div>
</template>

<script>
import { Pie } from "@antv/g2plot";

export default {
  props: {
    pieData: Array,
  },
  watch: {
    pieData: (pieData) => {
      if (pieData.length > 0) {
        const piePlot = new Pie("chart", {
          appendPadding: 10,
          data: pieData,
          angleField: "value",
          colorField: "name",
          radius: 1,
          innerRadius: 0.64,
          meta: {
            value: {
              formatter: (v) => `¥ ${v}`,
            },
          },
          label: {
            type: "inner",
            offset: "-50%",
            autoRotate: false,
            style: { textAlign: "center", fontSize: 16 },
            formatter: ({ percent }) => `${(percent * 100).toFixed(0)}%`,
          },
          legend: {
            itemName: { style: { fontSize: 14 } },
          },
          statistic: {
            title: {
              offsetY: -8,
            },
            content: {
              offsetY: -4,
            },
          },
          interactions: [
            { type: "element-selected" },
            { type: "element-active" },
            {
              type: "pie-statistic-active",
              cfg: {
                start: [
                  {
                    trigger: "element:mouseenter",
                    action: "pie-statistic:change",
                  },
                  {
                    trigger: "legend-item:mouseenter",
                    action: "pie-statistic:change",
                  },
                ],
                end: [
                  {
                    trigger: "element:mouseleave",
                    action: "pie-statistic:reset",
                  },
                  {
                    trigger: "legend-item:mouseleave",
                    action: "pie-statistic:reset",
                  },
                ],
              },
            },
          ],
        });
        piePlot.render();
      }
    },
  },
  mounted() {},
  methods() {},
};
</script>

<style scoped>
.pie {
  width: calc(50vw - 32px);
  box-sizing: border-box;
  box-shadow: #435a6f 0px 0px 1px 0px;
  padding: 16px 24px 24px;
}
.title {
  font-weight: 500;
  text-align: left;
}
#chart {
  height: 490px;
}
</style>
