// mongo_patch_models.js
// Run via: mongosh <your_database_name> mongo_patch_models.js

print("Starting MongoDB Models Patch Script...");

const dbName = db.getName();
print("Connected to database:", dbName);

// Find documents missing any of the required fields
const query = {
    $or: [
        { user_id: { $exists: false } },
        { dataset_name: { $exists: false } },
        { target_column: { $exists: false } },
        { metrics: { $exists: false } },
        { trained_at: { $exists: false } }
    ]
};

const cursor = db.models.find(query);
const modelsToPatch = cursor.toArray();

print(`Found ${modelsToPatch.length} documents requiring patching.`);

let patchedCount = 0;
let errorsCount = 0;

modelsToPatch.forEach(model => {
    try {
        let setUpdates = {};

        // 1. Backfill mapping data structurally ONLY if missing
        if (model.dataset_id) {
            const dataset = db.datasets.findOne({ _id: model.dataset_id });

            if (dataset) {
                if (model.user_id === undefined) {
                    setUpdates.user_id = dataset.user_id;
                }
                if (model.dataset_name === undefined) {
                    setUpdates.dataset_name = dataset.filename || dataset.file_name || "Unknown Dataset";
                }
                if (model.target_column === undefined) {
                    setUpdates.target_column = dataset.target_column;
                }
            } else {
                print(`[${model._id}] WARNING: Linked dataset not found.`);
            }
        }

        // 2. Patch missing base schema fields with null safely
        if (model.metrics === undefined) {
            setUpdates.metrics = null;
        }
        if (model.trained_at === undefined) {
            setUpdates.trained_at = null;
        }

        // Only update if there are fields to set
        if (Object.keys(setUpdates).length > 0) {
            db.models.updateOne(
                { _id: model._id },
                { $set: setUpdates }
            );
            patchedCount++;
            print(`[${model._id}] Patched fields: ${Object.keys(setUpdates).join(', ')}`);
        }
    } catch (e) {
        print(`[${model._id}] ERROR patching document: ${e.message}`);
        errorsCount++;
    }
});

print("\n--- MongoDB Patching Complete ---");
print(`Documents correctly patched: ${patchedCount}`);
print(`Errors encountered: ${errorsCount}`);
