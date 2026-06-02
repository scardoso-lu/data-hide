import type { NextConfig } from "next"

const nextConfig: NextConfig = {
  // Produce a minimal self-contained build for Docker.
  // The standalone output copies only the required node_modules so the
  // final image does not need to re-install dependencies.
  output: "standalone",
}

export default nextConfig
