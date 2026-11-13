import DefaultTheme from 'vitepress/theme'
import 'katex/dist/katex.min.css'
import 'markdown-it-texmath/css/texmath.css'
import './custom.css'
import TpFsdpDemo from '../components/TpFsdpDemo.vue'
import RingAllGatherDemo from '../components/RingAllGatherDemo.vue'
import AllToAllMatrixDemo from '../components/AllToAllMatrixDemo.vue'
import ReducerFlowDemo from '../components/ReducerFlowDemo.vue'

export default {
  ...DefaultTheme,
  enhanceApp({ app }) {
    DefaultTheme.enhanceApp?.({ app })
    app.component('TpFsdpDemo', TpFsdpDemo)
    app.component('RingAllGatherDemo', RingAllGatherDemo)
    app.component('AllToAllMatrixDemo', AllToAllMatrixDemo)
    app.component('ReducerFlowDemo', ReducerFlowDemo)
  },
}
