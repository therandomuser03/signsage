# text-to-sign (English -> ISL avatar)

English text -> gloss IDs (`remoteGlossMapper` via `apiClient.avatarGloss`, or `localGlossMapper`) -> pre-authored
clips on one VRM avatar (`AvatarViewer` + `useSignAnimation`). No generative model involved.

Assets you still have to create by hand (none are included):
- `public/models/avatar/signsage.vrm`   (VRoid Studio)
- `public/models/avatar/signs/<gloss>.glb`  one animation-only clip per sign, authored on the SAME rig (Blender)
- `public/models/avatar/manifest.json` lists them; its keys must equal the backend's `app/data/sign_vocabulary.json`
  (`tests/contract` fails if they differ).

Wiring:
```tsx
const { perform } = useSignAnimation(avatarRef, manifest, (text, m) => remoteGlossMapper(text, m, apiClient));
```
Manifest URLs use the `/models/avatar/...` prefix (`animationUrl` in manifest.json).
