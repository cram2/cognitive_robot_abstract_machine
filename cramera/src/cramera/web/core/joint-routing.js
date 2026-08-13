/* ============================================================================
 * core/joint-routing.js — which loaded joint a recorded connection name drives.
 *
 * A frame is keyed by the world's own connection name; a loaded model's joints carry the
 * names its URDF declares, under whatever prefix the bundler derived from its *bodies*.
 * Those namings need not agree. A world-merged MJCF robot has bodies called
 * "montessori/link0", so its model prefix is "montessori", while its connections stay
 * bare ("joint1") or start with a slash ("/finger_joint1"); a Gazebo world prefixes both.
 * So a key is tried as a prefixed name first -- that is what keeps two models declaring
 * the same joint name apart -- and then as a plain joint name across every model.
 * ==========================================================================*/
(function (global) {
  'use strict';

  function jointNamed(model, name) {
    const joints = model.obj && model.obj.joints;
    return (joints && joints[name]) || null;
  }

  function anyJointNamed(models, name) {
    for (let index = 0; index < models.length; index += 1) {
      const joint = jointNamed(models[index], name);
      if (joint) return joint;
    }
    return null;
  }

  global.JointRouting = {
    /* The joint `key` drives, or null when no loaded model declares it.

       `models` are the loaded model entries the shell holds: a `prefix` and the URDF's
       joints under `obj.joints`. */
    jointFor: function (models, key) {
      // a leading slash is part of the name, not an empty prefix
      const cut = key.indexOf('/');
      if (cut > 0) {
        const prefix = key.slice(0, cut);
        const name = key.slice(cut + 1);
        for (let index = 0; index < models.length; index += 1) {
          if (models[index].prefix !== prefix) continue;
          const joint = jointNamed(models[index], name);
          if (joint) return joint;
        }
        return anyJointNamed(models, key) || anyJointNamed(models, name);
      }
      return anyJointNamed(models, key);
    },
  };
})(window);
