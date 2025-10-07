async user_attendee_list_new(req, res) {
  try {
    const userId = new mongoose.Types.ObjectId(req.user_id);
    const start = parseInt(req.query.start) || 0;
    const limit = parseInt(req.query.limit) || 50;

    // Get profile info
    let profile_completed = false;
    let delegate_pass = false;
    const loginInfo = await login_info.findOne({ _id: userId }).lean();
    if (loginInfo) profile_completed = loginInfo.profile_percentage === 100;
    const pass = await attendee_passes.findOne({ email: loginInfo?.email_id }).sort({ _id: -1 }).lean();
    delegate_pass = pass && ["TNGSS Conference", "Conference"].includes(pass?.pass_type);

    let pipeline = [
      // AI Score
      {
        $match: {
          user_id: userId,
          score: { $exists: true, $gt: 0 }
        }
      },
      
      // Sort 
      { $sort: { score: -1 } },
      
      // pagination
      { $skip: start },
      { $limit: limit },
      
      // lookup user
      {
        $lookup: {
          from: "login_info",
          let: { matchedUserId: "$matched_user_id" },
          pipeline: [
            {
              $match: {
                $expr: { $eq: ["$_id", "$$matchedUserId"] },
                is_deleted: false,
                is_email_verified: true,
                role: "user"
              }
            }
          ],
          as: "user_info"
        }
      },
      
      // filter
      { $unwind: { path: "$user_info", preserveNullAndEmptyArrays: false } },
      
      // lookup attendee passes
      {
        $lookup: {
          from: "attendee-passes",
          let: { email: "$user_info.email_id" },
          pipeline: [
            { $match: { $expr: { $eq: ["$email", "$$email"] } } },
            ...(req.query.organisation_type ? [{ $match: { organisation_type: req.query.organisation_type } }] : []),
            ...(req.query.sector_interested ? [{ $match: { sector_interested: req.query.sector_interested } }] : []),
            { $project: { organisation_type: 1, sector_interested: 1 } },
            { $limit: 1 }
          ],
          as: "attendee_passes_details"
        }
      },
      
      // lookup profile managerment
      {
        $lookup: {
          from: "profile_management",
          let: { userId: "$user_info._id" },
          pipeline: [
            { $match: { $expr: { $eq: ["$user_id", "$$userId"] } } },
            ...(req.query.profile_type ? [{ $match: { profile_type: req.query.profile_type } }] : []),
            ...(req.query.focused_sector ? [{ $match: { focused_sector: req.query.focused_sector } }] : []),
            { $project: { designation: 1, organization_name: 1, bio: 1, profile_type: 1, focused_sector: 1 } },
            { $limit: 1 }
          ],
          as: "profile_details"
        }
      },
      
      // Lookup connection 
      {
        $lookup: {
          from: "connection_request_management",
          let: { targetUserId: "$user_info._id", currentUserId: userId },
          pipeline: [
            {
              $match: {
                $expr: {
                  $and: [
                    { $or: [{ $eq: ["$request_from", "$$currentUserId"] }, { $eq: ["$request_to", "$$currentUserId"] }] },
                    { $or: [{ $eq: ["$request_from", "$$targetUserId"] }, { $eq: ["$request_to", "$$targetUserId"] }] },
                    { $ne: ["$status", "declined"] },
                    { $eq: ["$is_deleted", false] }
                  ]
                }
              }
            },
            { $sort: { _id: -1 } },
            { $limit: 1 }
          ],
          as: "connection_info"
        }
      },
      
      // project
      {
        $project: {
          user_id: "$user_info._id",
          first_name: { $ifNull: ["$user_info.first_name", ""] },
          profile_image: "$user_info.profile_image",
          designation: { $ifNull: [{ $arrayElemAt: ["$profile_details.designation", 0] }, ""] },
          organization_name: { $ifNull: [{ $arrayElemAt: ["$profile_details.organization_name", 0] }, ""] },
          profile_percentage: { $ifNull: ["$user_info.profile_percentage", 0] },
          bio_exists: { $ne: [{ $ifNull: [{ $arrayElemAt: ["$profile_details.bio", 0] }, ""] }, ""] },
          already_connection_request_send: { $ne: [{ $arrayElemAt: ["$connection_info._id", 0] }, null] },
          connection_request_status: { $ifNull: [{ $arrayElemAt: ["$connection_info.status", 0] }, ""] },
          connection_reference_id: { $ifNull: [{ $arrayElemAt: ["$connection_info._id", 0] }, ""] },
          sender_or_reciever: {
            $switch: {
              branches: [
                { case: { $eq: [{ $toString: { $arrayElemAt: ["$connection_info.request_from", 0] } }, req.user_id] }, then: "sender" },
                { case: { $ne: [{ $arrayElemAt: ["$connection_info.request_from", 0] }, null] }, then: "receiver" }
              ],
              default: ""
            }
          },
          organisation_type: { $arrayElemAt: ["$attendee_passes_details.organisation_type", 0] },
          sector: { $arrayElemAt: ["$attendee_passes_details.sector_interested", 0] },
          profile_type: { $arrayElemAt: ["$profile_details.profile_type", 0] },
          focused_sector: { $arrayElemAt: ["$profile_details.focused_sector", 0] },
          ai_enabled: { $and: [delegate_pass, profile_completed, { $eq: ["$user_info.profile_percentage", 100] }] },
          ai_score: "$score"
        }
      }
    ];

    const attendees = await mongoose.connection
      .collection("user_ai_score")
      .aggregate(pipeline)
      .allowDiskUse(true)
      .toArray();

    const total_count = await mongoose.connection
      .collection("user_ai_score")
      .countDocuments({ user_id: userId, score: { $exists: true, $gt: 0 } });

    return jsend(200, "Successfully retrieved the attendees list", {
      total_count,
      attendee_list: attendees
    });

  } catch (e) {
    return jsend(406, e.message);
  }
}
