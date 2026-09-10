// Custom SIREN (Sinusoidal Representation Network) layer for TensorFlow.js.
//
// `sin` is NOT a built-in dense activation in tf.js, and SIREN needs a special
// init or the sines never leave their flat region:
//   - first layer:   W ~ U(±1/fan_in),          forward sin(ω₀·(Wx+b))
//   - hidden layers: W ~ U(±√(6/fan_in)/ω₀),    forward sin(ω₀·(Wx+b))
// (matches the reference implementation: ω₀ scales every layer's pre-activation)
//
// Scoping note: the class is created inside a tf-guard (so the page still
// loads when the CDN fails) and handed out through a closure variable — NOT
// by name. A `class` declared inside `if { ... }` is block-scoped, so the old
// `typeof SirenDense` check in sirenDense() was always "undefined" and every
// SIREN model silently fell back to a plain tanh layer. Never again.

const SIREN_W0 = 30;

let makeSirenLayer = null; // set just below, only if tf loaded

if (typeof tf !== 'undefined' && tf.layers && tf.layers.Layer) {
  class SirenDense extends tf.layers.Layer {
    static className = 'SirenDense';
    constructor(config) {
      super(config || {});
      this.units = config.units;
      this.isFirstLayer = config.isFirstLayer != null ? config.isFirstLayer : false;
      this.w0 = config.w0 != null ? config.w0 : SIREN_W0;
      this.useBias = config.useBias != null ? config.useBias : true;
    }
    build(inputShape) {
      const fanIn = inputShape[inputShape.length - 1];
      const bound = this.isFirstLayer
        ? 1 / Math.max(1, fanIn)
        : Math.sqrt(6 / Math.max(1, fanIn)) / this.w0;
      this.kernel = this.addWeight('kernel', [fanIn, this.units], 'float32',
        tf.initializers.randomUniform({ minval: -bound, maxval: bound }));
      if (this.useBias) {
        this.bias = this.addWeight('bias', [this.units], 'float32', tf.initializers.zeros());
      }
      this.built = true;
    }
    call(inputs) {
      return tf.tidy(() => {
        const x = Array.isArray(inputs) ? inputs[0] : inputs;
        let out = x.matMul(this.kernel.read());
        if (this.bias) out = out.add(this.bias.read());
        return tf.sin(tf.mul(out, this.w0));
      });
    }
    computeOutputShape(inputShape) {
      const s = inputShape.slice();
      s[s.length - 1] = this.units;
      return s;
    }
    getConfig() {
      const c = super.getConfig();
      c.units = this.units;
      c.isFirstLayer = this.isFirstLayer;
      c.w0 = this.w0;
      c.useBias = this.useBias;
      return c;
    }
  }
  try {
    if (tf.serialization && tf.serialization.registerClass) tf.serialization.registerClass(SirenDense);
  } catch (_) {}
  makeSirenLayer = (units, isFirstLayer, w0) => new SirenDense({
    units,
    w0: w0 != null ? w0 : SIREN_W0,
    isFirstLayer: !!isFirstLayer,
  });
}

function sirenDense(units, isFirstLayer, w0) {
  if (makeSirenLayer) return makeSirenLayer(units, isFirstLayer, w0);
  // fallback: regular dense with tanh if tf failed to load — still usable
  return tf.layers.dense({ units, activation: 'tanh' });
}
