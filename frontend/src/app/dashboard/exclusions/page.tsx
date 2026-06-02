import { getExclusions } from "@/lib/queries/exclusions"
import ExclusionTable from "./exclusion-table"

export const dynamic = "force-dynamic"

export default async function ExclusionsPage() {
  const exclusions = await getExclusions()

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-bold">Column Exclusions</h1>
        <p className="text-base-content/60 text-sm mt-1">
          Columns listed here are skipped by all anonymization tiers and passed
          through unchanged. Stored in{" "}
          <code className="font-mono text-xs bg-base-200 px-1 rounded">
            pii_column_exclusions
          </code>
          . Changes take effect on the next pipeline run.
        </p>
      </div>

      <div className="card bg-base-100 shadow p-6">
        <ExclusionTable exclusions={exclusions} />
      </div>
    </div>
  )
}
